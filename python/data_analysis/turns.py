"""Temporal turns: do zig-zag paths explain the connected-but-unentangled gap?

The spacetime graph is undirected, so a path joining two final-slice endpoints
may run *backward* in time, cross a gate, and run forward again. With
``MIPT_DIST_TURN_DIAGNOSTICS=1`` -- or offline, ``dist_graph_replay.exe augment
<stem>_records.bin``, which regenerates every trajectory's graph from the
master seed and writes ``<stem>_turns_records.bin`` -- each k=2 record carries
the *shape* of the pair's two canonical paths (shortest-maximin ``sm_*`` and
widest ``wd_*``): temporal turns, layer span, path class, bridges and how many
are measured or trace-only, and how many of the path's gates the doubled
network cancels; the fewest turns any walk needs (``min_turns``); and three
refined percolation events as flag bits,

    C_common_source (bit 7) <= C_certified (bit 8) <= C_doubled (bit 9) <= C (bit 4).

The last is the definitive one. A gate outside the past light cones of every
measurement and of both endpoints cancels against its adjoint in the density
matrix (U^dag U = I), and a pair the surviving network does not connect is an
exact product state -- a theorem, not a heuristic. The definitions live in the
"Temporal turns" and "The doubled network" comment blocks of
cpp/include/mipt/dist_connectivity.hpp; nothing here re-derives them.

:func:`temporal_turn_test` asks the question four ways, one output section each:

``conditional``   P(gap | N_turn = k) for N_turn in {sm_turns, wd_turns,
                  min_turns}, every observed k plus a pooled k >= 5 bucket; the
                  turn distributions of gap and of entangled connected pairs;
                  and P(gap | cancelled_gates > 0) for each canonical path.
``class``         P(gap | path class) and P(gap | trace-only bridge) for both
                  canonical paths, and the class mix of gap pairs (overall and
                  per zero class) against entangled connected pairs.
``matched``       every gap pair against ONE entangled connected control from
                  the *same trajectory*: same separation, same component size,
                  nearest shortest path, with recorded fallbacks. Separation
                  and component size both move every path statistic, so an
                  unmatched comparison would report geometry as a turn effect.
                  Distributions, class and trace-only fractions, matched
                  differences, and P(gap | N_turn) inside the matched sample,
                  whose baseline is exactly 1/2 by construction.
``connectivity``  kappa_D = P(E | C_D) and eta_D = P(E | not C_D) for each
                  definition D, overall and per separation, with kappa_D -
                  kappa_original and eta_D - eta_original on the same bootstrap
                  weights; the fraction of gap pairs each refinement *explains*
                  (C but not C_D) and the fraction of entangled connected pairs
                  it *costs* (they leave C_D, so they raise eta_D).

A **gap** pair is ``connected`` and not entangled -- not ``fn_positive`` by
default, which is exactly the run's own ``fmn > pair_zero_tol``; or ``fmn <=
zero_tol`` when a tolerance is passed. The zero class of a gap pair is the C++
``classify_unentangled`` rule, via :func:`pair_gap.zero_classes`.

Rules that are not negotiable:

* **NaN is never zero.** Every turn observable is NaN unless the pair is
  connected and the shape is available. A NaN is dropped from that
  observable's statistics and counted on a ``nan`` row of its own.
* **Errors are clustered on realization_id**, with the Poisson bootstrap of
  :mod:`pair_gap`: every pair of a trajectory shares one circuit and one
  measurement record, so per-record errors would be far too small. All CIs are
  95% percentile intervals over ``n_boot`` resamples.
* **Streamed, never loaded whole.** A turn-augmented record is 328 bytes, so
  the production file is ~6 GB on a 3.6 GB machine. One pass reduces the
  population sections to per-trajectory cell counts, folded into the bootstrap
  as they stream (each trajectory's Poisson weights are a pure function of
  ``(seed, realization_id)``, so the result is independent of the chunking);
  only the gap pairs and their matched controls are kept.

Outputs, beside ``<stem>_records.bin`` (``<stem>`` strips that suffix):

* ``<stem>_turn_summary.csv`` -- tidy long,
  ``section,metric,statistic,group,key,value,ci_low,ci_high,n``. ``key`` is the
  conditioning value (a turn count, ``>=5``, a class name, a separation,
  ``all`` or ``nan``) and ``n`` the denominator. A ``meta`` section records the
  tolerance, ``n_boot``, the seed and the record counts the rest rests on.
* ``<stem>_turn_selection.csv`` -- every gap pair and its matched control, for
  ``dist_graph_replay.exe paths``; header exactly :data:`SELECTION_HEADER`,
  sorted by realization, ``group`` in {gap, control}, each pair once.

``python -m data_analysis.turns <records.bin>`` runs it from the shell.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .pair_gap import (
    ZERO_CLASSES,
    _clustered_draws,
    _interval,
    _weighted_quantile_draws,
    gate_path_output_path,
    gate_record_context,
    iter_realization_blocks,
    zero_classes,
)
from .records import (
    FLAG_CONNECTED,
    FLAG_CONNECTED_CERTIFIED,
    FLAG_CONNECTED_COMMON_SOURCE,
    FLAG_CONNECTED_DOUBLED,
    FLAG_EVALUATED,
    FLAG_FN_POSITIVE,
    flag_bit_masks,
    record_file_info,
)
from .select_gate_traces import _embedding_sites, _format

# records::turn_observable_names() in dist_records.hpp, in payload order.
TURN_OBSERVABLES = (
    "sm_turns",
    "sm_temporal_forward",
    "sm_temporal_backward",
    "sm_min_layer",
    "sm_max_layer",
    "sm_class",
    "sm_bridges",
    "sm_bridges_measured",
    "sm_bridges_trace_only",
    "sm_cancelled_gates",
    "wd_turns",
    "wd_temporal_forward",
    "wd_temporal_backward",
    "wd_min_layer",
    "wd_max_layer",
    "wd_class",
    "wd_bridges",
    "wd_bridges_measured",
    "wd_bridges_trace_only",
    "wd_cancelled_gates",
    "min_turns",
)

# int(PathClass) -> path_class_name(), and the other two name tables of
# dist_connectivity.hpp, for the paths tool's text columns.
PATH_CLASSES = ("time_monotone", "common_source", "zigzag_certified", "zigzag_trace_only")
TRACE_ONLY_CLASS = 3
CONNECTOR_STATUSES = ("no_bridge", "all_measured", "mixed", "unmeasured")
BRIDGE_STATUSES = ("measured", "endpoint_anchored", "trace_only")

PATH_PREFIXES = ("sm", "wd")
TURN_METRICS = ("sm_turns", "wd_turns", "min_turns")
CLASS_METRICS = ("sm_class", "wd_class")
# P(gap | N_turn >= TURN_BUCKET) pools the tail, where single values are thin.
TURN_BUCKET = 5
TURN_BUCKET_KEY = f">={TURN_BUCKET}"

# Per-path counts compared as means between gap pairs and their controls.
MEAN_METRICS = tuple(
    f"{prefix}_{name}"
    for prefix in PATH_PREFIXES
    for name in (
        "bridges",
        "bridges_measured",
        "bridges_trace_only",
        "cancelled_gates",
        "temporal_forward",
        "min_layer",
    )
)

# (label, flag name, mask), the original percolation event first. Each later one
# refines the one before it, and the stream refuses a file where they do not nest.
CONNECTIVITY_DEFINITIONS = (
    ("original", "connected", FLAG_CONNECTED),
    ("common_source", "connected_common_source", FLAG_CONNECTED_COMMON_SOURCE),
    ("certified", "connected_certified", FLAG_CONNECTED_CERTIFIED),
    ("doubled", "connected_doubled", FLAG_CONNECTED_DOUBLED),
)
REFINED_FLAGS = tuple(name for _, name, _ in CONNECTIVITY_DEFINITIONS[1:])

# How a gap pair's control was found, best first: same geometry and the same
# component size (nearest shortest path among those) is the exact level.
TURN_MATCH_LEVELS = ("geometry_component_size", "geometry", "nearest_separation", "none")

SELECTION_COLUMNS = (
    "realization_id",
    "pair_i",
    "pair_j",
    "group",
    "geometry_id",
    "embedding_id",
    "separation",
    "match_level",
    "component_size",
    "shortest_path",
    "fmn",
)
SELECTION_HEADER = ",".join(SELECTION_COLUMNS)
SELECTION_GROUPS = ("gap", "control")

SUMMARY_COLUMNS = ("section", "metric", "statistic", "group", "key", "value", "ci_low", "ci_high", "n")

_TURN_DIAGNOSTICS_HINT = (
    "Produce them offline with `dist_graph_replay.exe augment <stem>_records.bin` (it writes "
    "<stem>_turns_records.bin beside the input), or rerun dist_scaling.exe (circ_type 2 or 4, "
    "k=2 or k=0) with MIPT_DIST_TURN_DIAGNOSTICS=1, which needs MIPT_DIST_GATE_DIAGNOSTICS=1."
)

# Stream tags, so the population weights, the matching tiebreaks and the
# matched-sample bootstrap never share a random stream.
_POPULATION_STREAM = 13
_MATCH_STREAM = 11
_MATCHED_SEED_OFFSET = 29

# A turn-augmented record is 328 bytes, so the default million-record chunk would
# be 328 MB -- and iter_realization_blocks holds up to three chunks at once while
# it carries a trajectory across a boundary. The chunk is therefore capped in
# bytes. Results do not depend on it (they are chunk-invariant by construction,
# and tested so); only memory and speed do.
_CHUNK_BYTE_BUDGET = 96 << 20
# Whole trajectories are reduced this many records at a time.
_POPULATION_BATCH_RECORDS = 1 << 16

# Poisson(1) by inverse CDF. P(K > 31) ~ 1e-35, so the table is complete in
# double precision.
_POISSON_CDF = np.cumsum([math.exp(-1.0) / math.factorial(k) for k in range(32)])

# Per-geometry connectivity cells: evaluated, entangled, then per definition
# C_D, C_D & E, gap & !C_D, (C & E) & !C_D.
_GEOMETRY_CELLS = 2 + 4 * len(CONNECTIVITY_DEFINITIONS)


def turn_output_path(records: str | Path, suffix: str) -> Path:
    """``<stem><suffix>`` beside a ``<stem>_records.bin``."""
    return gate_path_output_path(records, suffix)


def turn_record_context(records: str | Path, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a record file for :func:`temporal_turn_test` and collect what it needs.

    Refuses, naming the two ways to produce them, a file without the 21 turn
    observables, the header's ``turn_diagnostics=1``, or flag bits 7-9 declared
    on its ``flag_bits=`` line. Everything the gate-path context checks (k=2,
    connectivity flags, the gate observables the matching uses) is checked too.
    """
    info = info or record_file_info(records)
    if int(info["k"]) != 2:
        raise ValueError(f"{records}: temporal turns are per pair; this is a k={info['k']} file.")
    header = info["header"]
    declared = flag_bit_masks(header) if info["has_flags"] else {}
    missing = []
    if str(header.get("turn_diagnostics", "")).strip() != "1":
        missing.append("the header's turn_diagnostics=1")
    present = set(info["observables"])
    observables = [name for name in TURN_OBSERVABLES if name not in present]
    if observables:
        missing.append(f"{len(observables)} of the {len(TURN_OBSERVABLES)} turn observables ({', '.join(observables[:4])}{', ...' if len(observables) > 4 else ''})")
    bits = [name for name in ("connected",) + REFINED_FLAGS if name not in declared]
    if bits:
        missing.append(f"flag bits {', '.join(bits)}")
    if missing:
        raise ValueError(
            f"{records}: no temporal-turn diagnostics -- missing {'; '.join(missing)}. "
            + _TURN_DIAGNOSTICS_HINT
        )
    context = gate_record_context(records, info)
    try:
        version = int(header.get("turn_diagnostics_version"))
    except (TypeError, ValueError):
        version = -1
    context.update(
        {
            "turn_diagnostics_version": version,
            "turn_diagnostics_definition": header.get("turn_diagnostics_definition"),
            "declared_flags": declared,
        }
    )
    return context


# ---------------------------------------------------------------------------
# The streamed population reduction (sections conditional, class, connectivity)
# ---------------------------------------------------------------------------


def _trajectory_weights(realizations: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Poisson(1) bootstrap weights, shaped ``(n_boot, len(realizations))``.

    Column r is a pure function of ``(seed, realization_id)``: trajectory r's
    weights are the inverse-CDF images of the uniforms at positions
    ``[r n_boot, (r+1) n_boot)`` of one PCG64 stream, reached with ``advance``
    (one 64-bit draw per uniform, so the position is exact). That is what makes
    a bootstrap folded in batch by batch independent of how the file was
    chunked and of which trajectories a strided file kept. Drawing per batch
    from one running generator would tie every weight to the batch boundaries.
    """
    out = np.empty((n_boot, realizations.size))
    if n_boot == 0 or realizations.size == 0:
        return out
    sequence = np.random.SeedSequence([int(seed), _POPULATION_STREAM])
    breaks = np.flatnonzero(np.diff(realizations) != 1) + 1
    position = 0
    for run in np.split(realizations, breaks):
        bits = np.random.PCG64(sequence)
        bits.advance(int(run[0]) * n_boot)
        uniform = np.random.Generator(bits).random((run.size, n_boot))
        out[:, position : position + run.size] = np.searchsorted(_POISSON_CDF, uniform, side="right").T
        position += run.size
    return out


class _Streamed:
    """Cell counts folded in trajectory by trajectory: exact totals and bootstrap sums.

    ``boot[b] = sum_r w_br counts_r``. Weights and counts are both integers, so
    every partial sum is an exact integer in float64 and the grouping of the
    trajectories into batches cannot change a single bit of the result.
    """

    def __init__(self, n_boot: int, width: int):
        self.total = np.zeros(width, dtype=np.int64)
        self.boot = np.zeros((n_boot, width))

    @property
    def width(self) -> int:
        return int(self.total.size)

    def grow(self, width: int) -> None:
        # New cells are appended, so a cell's index never moves, and every
        # trajectory already folded in had zero counts in them.
        extra = int(width) - self.width
        if extra > 0:
            self.total = np.concatenate([self.total, np.zeros(extra, dtype=np.int64)])
            self.boot = np.concatenate([self.boot, np.zeros((self.boot.shape[0], extra))], axis=1)

    def add(self, weights: np.ndarray, flat: np.ndarray, clusters: int) -> None:
        if flat.size == 0:
            return
        counts = np.bincount(flat, minlength=clusters * self.width).reshape(clusters, self.width)
        self.total += counts.sum(axis=0)
        if self.boot.shape[0]:
            self.boot += weights @ counts.astype(np.float64)

    def vector(self, plus: Sequence[int] = (), minus: Sequence[int] = (), weights=None) -> np.ndarray:
        out = np.zeros(self.width)
        cells = np.asarray(list(plus), dtype=np.int64)
        np.add.at(out, cells, 1.0 if weights is None else np.asarray(weights, dtype=float))
        np.add.at(out, np.asarray(list(minus), dtype=np.int64), -1.0)
        return out

    def ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, np.ndarray, float]:
        """sum(numerator . cells) / sum(denominator . cells): point, draws, denominator."""
        top = float(self.total @ numerator)
        bottom = float(self.total @ denominator)
        value = top / bottom if bottom > 0 else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (self.boot @ numerator) / (self.boot @ denominator)
        return value, draws, bottom


def _integer_values(values: np.ndarray, name: str, upper: int | None = None) -> np.ndarray:
    """Finite stored counts as int64, refusing anything that is not one."""
    integers = values.astype(np.int64)
    bad = (integers != values) | (integers < 0)
    if upper is not None:
        bad |= integers > upper
    if np.any(bad):
        raise ValueError(
            f"{name} holds {values[bad][0]!r}, which is not a valid "
            f"{'class code' if upper is not None else 'count'}; the record file is damaged "
            "or was written under a different turn-diagnostics definition."
        )
    return integers


class _TurnPopulation:
    """Sections conditional, class and connectivity, reduced as the records stream."""

    def __init__(self, context: dict[str, Any], *, n_boot: int, seed: int, zero_tol: float | None):
        self.context = context
        self.n_boot = n_boot
        self.seed = seed
        self.zero_tol = zero_tol
        self.entanglement = context["entanglement"]
        self.geometries = int(context["separation"].size)
        self.layout: dict[str, int] = {}
        width = 0
        zero_count = len(ZERO_CLASSES)
        for name, size in (
            ("connected", 1),
            ("gap", 1),
            ("gap_zero_class", zero_count),
            *(
                item
                for metric in CLASS_METRICS
                for item in (
                    (f"{metric}|connected", len(PATH_CLASSES)),
                    (f"{metric}|gap", len(PATH_CLASSES)),
                    (f"{metric}|gap_zero_class", zero_count * len(PATH_CLASSES)),
                    (f"{metric}|nan", 2),
                )
            ),
            *((f"{metric}|nan", 2) for metric in TURN_METRICS),
            *(
                item
                for prefix in PATH_PREFIXES
                for item in ((f"{prefix}_cancelled_gates|connected", 3), (f"{prefix}_cancelled_gates|gap", 3))
            ),
            ("geometry", self.geometries * _GEOMETRY_CELLS),
        ):
            self.layout[name] = width
            width += size
        self.fixed = _Streamed(n_boot, width)
        self.turns = {metric: _Streamed(n_boot, 2 * (TURN_BUCKET + 1)) for metric in TURN_METRICS}
        self.records = 0
        self.trajectories = 0

    def _check(self, batch, evaluated, connected, refined, ids) -> None:
        def refuse(mask, what):
            if np.any(mask):
                first = int(ids[np.flatnonzero(mask)[0]])
                raise ValueError(
                    f"{self.context['info']['path']}: {int(mask.sum())} record(s) {what} (first in "
                    f"realization {first}). The file contradicts the turn-diagnostics contract."
                )

        refuse(connected & ~evaluated, "are connected but not evaluated")
        chain = list(refined[1:]) + [connected]
        names = list(REFINED_FLAGS) + ["connected"]
        for inner, outer, inner_name, outer_name in zip(chain[:-1], chain[1:], names[:-1], names[1:]):
            refuse(inner & ~outer, f"are {inner_name} but not {outer_name}")
        for name in TURN_OBSERVABLES:
            refuse(np.isfinite(batch[name]) & ~connected, f"carry a finite {name} without being connected")

    def absorb(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        ids = batch["realization_id"].astype(np.int64)
        realizations, cluster = np.unique(ids, return_inverse=True)
        cluster = cluster.astype(np.int64)
        clusters = int(realizations.size)
        self.records += int(batch.size)
        self.trajectories += clusters
        weights = _trajectory_weights(realizations, self.n_boot, self.seed)

        flags = batch["flags"].astype(np.int64)
        evaluated = (flags & FLAG_EVALUATED) != 0
        refined = [(flags & mask) != 0 for _, _, mask in CONNECTIVITY_DEFINITIONS]
        connected = refined[0]
        if self.zero_tol is None:
            entangled = (flags & FLAG_FN_POSITIVE) != 0
        else:
            # NaN compares false, as in the C++ `entanglement > pair_zero_tol`.
            entangled = np.asarray(batch[self.entanglement], dtype=float) > self.zero_tol
        gap = connected & ~entangled
        entangled_connected = connected & entangled
        self._check(batch, evaluated, connected, refined, ids)

        zero_class = np.full(batch.size, -1, dtype=np.int64)
        gap_index = np.flatnonzero(gap)
        if gap_index.size and self.context["classifiable"]:
            names = zero_classes(batch[gap_index], self.context)
            for position, name in enumerate(ZERO_CLASSES):
                zero_class[gap_index[names == name]] = position

        layout = self.layout
        width = self.fixed.width
        base = cluster * width
        parts: list[np.ndarray] = []

        def put(mask: np.ndarray, cell) -> None:
            if np.isscalar(cell):
                parts.append(base[mask] + int(cell))
            else:
                parts.append(base[mask] + cell[mask])

        put(connected, layout["connected"])
        put(gap, layout["gap"])
        classified = gap & (zero_class >= 0)
        put(classified, layout["gap_zero_class"] + zero_class)
        classes = len(PATH_CLASSES)
        for metric in CLASS_METRICS:
            values = np.asarray(batch[metric], dtype=float)
            finite = np.isfinite(values)
            code = np.zeros(batch.size, dtype=np.int64)
            code[finite] = _integer_values(values[finite], metric, upper=classes - 1)
            put(connected & finite, layout[f"{metric}|connected"] + code)
            put(gap & finite, layout[f"{metric}|gap"] + code)
            put(classified & finite, layout[f"{metric}|gap_zero_class"] + zero_class * classes + code)
            put(connected & ~finite, layout[f"{metric}|nan"])
            put(gap & ~finite, layout[f"{metric}|nan"] + 1)
        for metric in TURN_METRICS:
            finite = np.isfinite(np.asarray(batch[metric], dtype=float))
            put(connected & ~finite, layout[f"{metric}|nan"])
            put(gap & ~finite, layout[f"{metric}|nan"] + 1)
        for prefix in PATH_PREFIXES:
            name = f"{prefix}_cancelled_gates"
            values = np.asarray(batch[name], dtype=float)
            finite = np.isfinite(values)
            state = np.full(batch.size, 2, dtype=np.int64)  # 0: none cancelled, 1: some, 2: NaN
            state[finite] = (_integer_values(values[finite], name) > 0).astype(np.int64)
            put(connected, layout[f"{name}|connected"] + state)
            put(gap, layout[f"{name}|gap"] + state)

        cell = layout["geometry"] + batch["geometry_id"].astype(np.int64) * _GEOMETRY_CELLS
        put(evaluated, cell)
        put(evaluated & entangled, cell + 1)
        for position, inside in enumerate(refined):
            offset = 2 + 4 * position
            put(inside, cell + offset)
            put(inside & entangled, cell + offset + 1)
            put(gap & ~inside, cell + offset + 2)
            put(entangled_connected & ~inside, cell + offset + 3)
        self.fixed.add(weights, np.concatenate(parts), clusters)

        for metric in TURN_METRICS:
            values = np.asarray(batch[metric], dtype=float)
            usable = connected & np.isfinite(values)
            if not np.any(usable):
                continue
            turns = _integer_values(values[usable], metric)
            table = self.turns[metric]
            table.grow(2 * max(int(turns.max()) + 1, TURN_BUCKET + 1))
            table.add(weights, cluster[usable] * table.width + 2 * turns + gap[usable], clusters)

        return {"ids": ids, "gap": gap, "entangled_connected": entangled_connected, "zero_class": zero_class}


# ---------------------------------------------------------------------------
# Matching (section matched)
# ---------------------------------------------------------------------------


def match_turn_controls(
    gaps: np.ndarray,
    controls: np.ndarray,
    separation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """For each gap pair, the index of its control in ``controls`` and the match level.

    ``gaps`` and ``controls`` are records of *one* trajectory. Key, most
    significant first: same geometry before any other; among other geometries
    the nearest separation; then the nearest component size (an exact match
    first); then the nearest shortest path; then a tiebreak drawn from ``rng``.
    Each gap pair picks independently, so a control can serve several (sampling
    with replacement, as :func:`pair_gap.gate_path_gap_test` does). Returns -1
    and ``none`` for every gap pair when the trajectory has no control. A NaN
    size or path is infinitely far from everything, never equal to it.
    """
    count = gaps.size
    if controls.size == 0:
        return np.full(count, -1, dtype=np.int64), np.full(count, "none", dtype=object)
    gap_geometry = gaps["geometry_id"].astype(np.int64)[:, None]
    control_geometry = controls["geometry_id"].astype(np.int64)[None, :]
    same = gap_geometry == control_geometry
    with np.errstate(invalid="ignore"):
        separation_gap = np.abs(separation[gap_geometry] - separation[control_geometry])
        size_gap = np.abs(
            np.asarray(gaps["component_size"], dtype=float)[:, None]
            - np.asarray(controls["component_size"], dtype=float)[None, :]
        )
        path_gap = np.abs(
            np.asarray(gaps["shortest_path"], dtype=float)[:, None]
            - np.asarray(controls["shortest_path"], dtype=float)[None, :]
        )
    separation_gap = np.where(same, 0.0, np.nan_to_num(separation_gap, nan=np.inf))
    size_gap = np.nan_to_num(size_gap, nan=np.inf)
    path_gap = np.nan_to_num(path_gap, nan=np.inf)
    tiebreak = rng.random(same.shape)
    order = np.lexsort((tiebreak, path_gap, size_gap, separation_gap, (~same).astype(np.int8)), axis=-1)
    best = order[:, 0]
    rows = np.arange(count)
    level = np.where(
        same[rows, best],
        np.where(size_gap[rows, best] == 0.0, TURN_MATCH_LEVELS[0], TURN_MATCH_LEVELS[1]),
        TURN_MATCH_LEVELS[2],
    ).astype(object)
    return best.astype(np.int64), level


class _TurnMatcher:
    """Matches gap pairs to controls trajectory by trajectory and keeps both."""

    def __init__(self, context: dict[str, Any], *, seed: int):
        self.context = context
        self.seed = seed
        self.separation = context["separation"]
        self.fields = (
            "geometry_id",
            "embedding_id",
            "flags",
            "component_size",
            "shortest_path",
            context["entanglement"],
            *TURN_OBSERVABLES,
        )
        self.gap = {name: [] for name in self.fields}
        self.control = {name: [] for name in self.fields}
        self.realization: list[np.ndarray] = []
        self.zero_class: list[np.ndarray] = []
        self.level: list[np.ndarray] = []
        self.gap_trajectories = 0

    def absorb(self, batch: np.ndarray, masks: dict[str, np.ndarray]) -> None:
        gap = masks["gap"]
        if not np.any(gap):
            return
        ids = masks["ids"]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(ids)) + 1, [ids.size]])
        per_trajectory = np.add.reduceat(gap.astype(np.int64), starts[:-1])
        controls_mask = masks["entangled_connected"]
        for position in np.flatnonzero(per_trajectory):
            begin, end = int(starts[position]), int(starts[position + 1])
            realization = int(ids[begin])
            block = batch[begin:end]
            here = gap[begin:end]
            gaps = block[here]
            controls = block[controls_mask[begin:end]]
            # Seeded per trajectory, so a match never depends on the chunking.
            rng = np.random.default_rng([self.seed, _MATCH_STREAM, realization])
            index, level = match_turn_controls(gaps, controls, self.separation, rng)
            self.gap_trajectories += 1
            self.realization.append(np.full(gaps.size, realization, dtype=np.int64))
            self.zero_class.append(masks["zero_class"][begin:end][here])
            self.level.append(level)
            for name in self.fields:
                self.gap[name].append(np.asarray(gaps[name]))
            if controls.size:
                picked = controls[index]
                for name in self.fields:
                    self.control[name].append(np.asarray(picked[name]))
            else:
                for name in self.fields:
                    fill = -1 if name in ("geometry_id", "embedding_id") else (0 if name == "flags" else np.nan)
                    self.control[name].append(np.full(gaps.size, fill, dtype=np.float64))

    def pairs(self, info: dict[str, Any]) -> pd.DataFrame:
        """One row per gap pair, with its matched control's identifiers and values."""
        entanglement = self.context["entanglement"]

        def joined(parts: list[np.ndarray], dtype=None) -> np.ndarray:
            if not parts:
                return np.empty(0, dtype=dtype or np.float64)
            return np.concatenate(parts).astype(dtype) if dtype else np.concatenate(parts)

        sites = _embedding_sites(info)
        shape = (max(g for g, _ in sites) + 1, max(e for _, e in sites) + 1)
        site_i = np.full(shape, -1, dtype=np.int64)
        site_j = np.full(shape, -1, dtype=np.int64)
        for (geometry, embedding), (first, second) in sites.items():
            site_i[geometry, embedding] = first
            site_j[geometry, embedding] = second

        def endpoints(geometry: np.ndarray, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            valid = (geometry >= 0) & (embedding >= 0)
            i = np.full(geometry.size, -1, dtype=np.int64)
            j = np.full(geometry.size, -1, dtype=np.int64)
            i[valid] = site_i[geometry[valid], embedding[valid]]
            j[valid] = site_j[geometry[valid], embedding[valid]]
            return i, j

        zero_names = np.array(list(ZERO_CLASSES) + ["unknown"], dtype=object)
        zero_class = joined(self.zero_class, np.int64)
        frame: dict[str, Any] = {"realization_id": joined(self.realization, np.int64)}
        for side, prefix in ((self.gap, ""), (self.control, "control_")):
            geometry = joined(side["geometry_id"], np.int64)
            embedding = joined(side["embedding_id"], np.int64)
            pair_i, pair_j = endpoints(geometry, embedding)
            frame[prefix + "geometry_id"] = geometry
            frame[prefix + "embedding_id"] = embedding
            frame[prefix + "pair_i"] = pair_i
            frame[prefix + "pair_j"] = pair_j
            if not prefix:
                safe = np.clip(geometry, 0, None)
                frame["separation"] = np.where(geometry >= 0, self.separation[safe], np.nan)
                frame["d"] = np.where(geometry >= 0, self.context["distance"][safe], np.nan)
                frame["zero_class"] = zero_names[np.where(zero_class >= 0, zero_class, len(ZERO_CLASSES))]
                frame["match_level"] = joined(self.level).astype(object) if self.level else np.empty(0, dtype=object)
            matched = geometry >= 0
            flags = joined(side["flags"], np.int64)
            for (_, name, mask) in CONNECTIVITY_DEFINITIONS[1:]:
                frame[prefix + name] = np.where(matched, ((flags & mask) != 0).astype(float), np.nan)
            for name in ("component_size", "shortest_path", entanglement, *TURN_OBSERVABLES):
                frame[prefix + name] = joined(side[name], np.float64)
        return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# The matched-sample bootstrap
# ---------------------------------------------------------------------------


class _ClusterBook:
    """Ratios of per-record sums with trajectory-clustered draws, all on one set of weights.

    Every numerator and denominator is reduced to per-trajectory sums once;
    :func:`pair_gap._clustered_draws` then resamples all of them together, so
    every statistic -- and every difference between a gap group and its
    controls, which share their trajectories -- is paired.
    """

    def __init__(self, realization: np.ndarray):
        ids, cluster = np.unique(np.asarray(realization, dtype=np.int64), return_inverse=True)
        self.cluster = cluster.astype(np.int64)
        self.clusters = max(int(ids.size), 1)
        self.columns: list[np.ndarray] = []
        self.totals: list[float] = []
        self.specs: list[tuple] = []
        self.draws: np.ndarray | None = None

    def column(self, values: np.ndarray) -> int:
        values = np.asarray(values, dtype=float)
        self.columns.append(np.bincount(self.cluster, weights=values, minlength=self.clusters))
        self.totals.append(float(values.sum()))
        return len(self.columns) - 1

    def ratio(self, meta: tuple, numerator: np.ndarray, denominator: np.ndarray) -> None:
        self.specs.append((meta, self.column(numerator), self.column(denominator)))

    def resolve(self, n_boot: int, seed: int) -> list[tuple]:
        rows = []
        if self.columns and n_boot:
            matrix = np.stack(self.columns, axis=1)
            draws = np.empty((n_boot, matrix.shape[1]))
            for where, weights in _clustered_draws(self.clusters, n_boot, seed):
                draws[where] = weights @ matrix
        else:
            draws = np.empty((0, len(self.columns)))
        for meta, top, bottom in self.specs:
            denominator = self.totals[bottom]
            value = self.totals[top] / denominator if denominator > 0 else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                ci = _interval(draws[:, top] / draws[:, bottom])
            rows.append((meta, value, ci, denominator))
        return rows


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def temporal_turn_test(
    records: str | Path,
    *,
    zero_tol: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    chunk_records: int = 100_000,
    write: bool = True,
    output: str | Path | None = None,
    selection_output: str | Path | None = None,
    show_summary: bool = True,
) -> dict[str, Any]:
    """Do zig-zag and cancelled paths explain the gap? The four-section test.

    ``records`` is a turn-augmented k=2 record binary (``*_turns_records.bin``
    from ``dist_graph_replay.exe augment``, or a dist_scaling.exe file written
    with ``MIPT_DIST_TURN_DIAGNOSTICS=1``); anything else is refused by name.
    ``zero_tol=None`` classifies entanglement by the run's own ``fn_positive``
    flag; a number re-thresholds ``fmn`` instead (``fmn > zero_tol``). See the
    module docstring for the sections, the NaN rule and the bootstrap.

    ``chunk_records`` bounds the read, and is itself capped at 96 MB of
    records; neither changes a number. Writes ``<stem>_turn_summary.csv``
    (``output``) and ``<stem>_turn_selection.csv`` (``selection_output``)
    unless ``write=False``.

    Returns a dict: ``summary`` (the tidy table) and its four sections
    ``conditional``, ``classes``, ``matched`` and ``connectivity``;
    ``kappa_eta`` (one row per connectivity definition, the overall numbers);
    ``pairs`` (one row per gap pair with its matched control); ``selection``;
    ``match_levels``; ``path``, ``selection_path``; and ``context``.
    """
    if int(chunk_records) <= 0:
        raise ValueError("chunk_records must be positive.")
    context = turn_record_context(records)
    info = context["info"]
    n_boot = max(int(n_boot), 0)
    tolerance = None if zero_tol is None else float(zero_tol)
    chunk = max(1, min(int(chunk_records), _CHUNK_BYTE_BUDGET // max(int(info["record_bytes"]), 1)))

    population = _TurnPopulation(context, n_boot=n_boot, seed=seed, zero_tol=tolerance)
    matcher = _TurnMatcher(context, seed=seed)
    pending: list[np.ndarray] = []
    pending_records = 0

    def flush() -> None:
        batch = pending[0] if len(pending) == 1 else np.concatenate(pending)
        matcher.absorb(batch, population.absorb(batch))

    for _, block in iter_realization_blocks(records, info=info, chunk_records=chunk):
        pending.append(block)
        pending_records += block.size
        if pending_records >= _POPULATION_BATCH_RECORDS:
            flush()
            pending.clear()
            pending_records = 0
    if pending:
        flush()
        pending.clear()

    rows: list[tuple] = []

    def add(section, metric, statistic, group, key, value, ci=(np.nan, np.nan), n=0.0) -> None:
        rows.append((section, metric, statistic, group, str(key), float(value), float(ci[0]), float(ci[1]), float(n)))

    _population_rows(population, add)
    pairs = matcher.pairs(info)
    match_levels = _matched_rows(pairs, context, n_boot=n_boot, seed=seed, add=add)
    kappa_eta = _connectivity_rows(population, add)

    reference = float(info.get("pair_zero_tol") or np.nan)
    add("meta", "zero_tol", "value", "fn_positive_flag" if tolerance is None else "explicit", "all",
        reference if tolerance is None else tolerance)
    add("meta", "pair_zero_tol", "value", "header", "all", reference)
    add("meta", "n_boot", "value", "all", "all", n_boot)
    add("meta", "seed", "value", "all", "all", seed)
    add("meta", "ci_level", "value", "all", "all", 0.95)
    add("meta", "turn_diagnostics_version", "value", "header", "all", context["turn_diagnostics_version"])
    add("meta", "records", "count", "all", "all", population.records, n=population.records)
    add("meta", "trajectories", "count", "all", "all", population.trajectories, n=population.trajectories)
    add("meta", "trajectories", "count", "gap", "all", matcher.gap_trajectories, n=population.trajectories)
    add("meta", "pairs", "count", "gap", "all", len(pairs), n=len(pairs))

    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    summary["n"] = summary["n"].astype(np.int64)
    selection = _selection_frame(pairs, context)

    path = selection_path = None
    if write:
        path = Path(output) if output is not None else turn_output_path(records, "_turn_summary.csv")
        summary.to_csv(path, index=False, float_format="%.17g")
        selection_path = (
            Path(selection_output)
            if selection_output is not None
            else turn_output_path(records, "_turn_selection.csv")
        )
        _write_selection(selection, selection_path)

    result = {
        "summary": summary,
        "conditional": summary.loc[summary["section"] == "conditional"].reset_index(drop=True),
        "classes": summary.loc[summary["section"] == "class"].reset_index(drop=True),
        "matched": summary.loc[summary["section"] == "matched"].reset_index(drop=True),
        "connectivity": summary.loc[summary["section"] == "connectivity"].reset_index(drop=True),
        "kappa_eta": kappa_eta,
        "pairs": pairs,
        "selection": selection,
        "match_levels": match_levels,
        "path": path,
        "selection_path": selection_path,
        "context": context,
    }
    if show_summary:
        _print_turn_summary(result, records)
    return result


def _population_rows(population: _TurnPopulation, add) -> None:
    """Sections conditional and class, from the streamed totals."""
    fixed = population.fixed
    layout = population.layout

    def emit(section, metric, statistic, group, key, table, numerator, denominator):
        value, draws, n = table.ratio(numerator, denominator)
        add(section, metric, statistic, group, key, value, _interval(draws), n)

    connected = layout["connected"]
    gap = layout["gap"]
    emit("conditional", "connected", "p_gap", "connected", "all", fixed, fixed.vector([gap]), fixed.vector([connected]))
    add("conditional", "connected", "count", "connected", "all", fixed.total[connected], n=fixed.total[connected])
    add("conditional", "connected", "count", "gap", "all", fixed.total[gap], n=fixed.total[connected])

    for metric in TURN_METRICS:
        table = population.turns[metric]
        values = np.arange(table.width // 2)
        conn = table.total[0::2] + table.total[1::2]
        observed = [int(k) for k in values[conn > 0]]
        tail = [int(k) for k in values if k >= TURN_BUCKET]
        selections = [(str(k), [k]) for k in observed] + [(TURN_BUCKET_KEY, tail), ("all", list(values))]
        for key, ks in selections:
            gaps = [2 * k + 1 for k in ks]
            entangled = [2 * k for k in ks]
            emit("conditional", metric, "p_gap", "connected", key, table, table.vector(gaps), table.vector(gaps + entangled))
            if key == "all":
                continue
            emit("conditional", metric, "fraction", "gap", key, table, table.vector(gaps), table.vector(2 * values + 1))
            emit("conditional", metric, "fraction", "entangled_connected", key, table,
                 table.vector(entangled), table.vector(2 * values))
        for group, offset in (("gap", 1), ("entangled_connected", 0)):
            cells = 2 * values + offset
            emit("conditional", metric, "mean", group, "all", table, table.vector(cells, weights=values), table.vector(cells))
        nan = layout[f"{metric}|nan"]
        emit("conditional", metric, "p_gap", "connected", "nan", fixed, fixed.vector([nan + 1]), fixed.vector([nan]))
        add("conditional", metric, "count", "connected", "nan", fixed.total[nan], n=fixed.total[connected])
        add("conditional", metric, "count", "gap", "nan", fixed.total[nan + 1], n=fixed.total[gap])

    for prefix in PATH_PREFIXES:
        name = f"{prefix}_cancelled_gates"
        conn = layout[f"{name}|connected"]
        gaps = layout[f"{name}|gap"]
        for key, offset in ((">0", 1), ("0", 0), ("nan", 2)):
            emit("conditional", name, "p_gap", "connected", key, fixed, fixed.vector([gaps + offset]),
                 fixed.vector([conn + offset]))
        # P(cancelled > 0 | group), among pairs whose count is known.
        emit("conditional", name, "fraction", "gap", ">0", fixed, fixed.vector([gaps + 1]), fixed.vector([gaps, gaps + 1]))
        emit("conditional", name, "fraction", "entangled_connected", ">0", fixed,
             fixed.vector([conn + 1], minus=[gaps + 1]), fixed.vector([conn, conn + 1], minus=[gaps, gaps + 1]))

    classes = len(PATH_CLASSES)
    for metric, prefix in zip(CLASS_METRICS, PATH_PREFIXES):
        conn = layout[f"{metric}|connected"]
        gaps = layout[f"{metric}|gap"]
        by_zero = layout[f"{metric}|gap_zero_class"]
        every = range(classes)
        for code, name in enumerate(PATH_CLASSES):
            emit("class", metric, "p_gap", "connected", name, fixed, fixed.vector([gaps + code]), fixed.vector([conn + code]))
        trace_only = [TRACE_ONLY_CLASS]
        others = [code for code in every if code != TRACE_ONLY_CLASS]
        for key, codes in (("trace_only", trace_only), ("not_trace_only", others)):
            emit("class", f"{prefix}_trace_only_bridge", "p_gap", "connected", key, fixed,
                 fixed.vector([gaps + c for c in codes]), fixed.vector([conn + c for c in codes]))
        for code, name in enumerate(PATH_CLASSES):
            emit("class", metric, "fraction", "gap", name, fixed, fixed.vector([gaps + code]),
                 fixed.vector([gaps + c for c in every]))
            emit("class", metric, "fraction", "entangled_connected", name, fixed,
                 fixed.vector([conn + code], minus=[gaps + code]),
                 fixed.vector([conn + c for c in every], minus=[gaps + c for c in every]))
            if population.context["classifiable"]:
                for position, zero_name in enumerate(ZERO_CLASSES):
                    start = by_zero + position * classes
                    emit("class", metric, "fraction", f"gap_{zero_name}", name, fixed,
                         fixed.vector([start + code]), fixed.vector([start + c for c in every]))
        nan = layout[f"{metric}|nan"]
        add("class", metric, "count", "connected", "nan", fixed.total[nan], n=fixed.total[connected])
        add("class", metric, "count", "gap", "nan", fixed.total[nan + 1], n=fixed.total[gap])
    if population.context["classifiable"]:
        for position, zero_name in enumerate(ZERO_CLASSES):
            cell = layout["gap_zero_class"] + position
            add("class", "zero_class", "count", "gap", zero_name, fixed.total[cell], n=fixed.total[gap])


def _connectivity_rows(population: _TurnPopulation, add) -> pd.DataFrame:
    """Section connectivity; returns the overall numbers as one row per definition."""
    fixed = population.fixed
    start = population.layout["geometry"]
    separation = population.context["separation"]
    present = [g for g in range(population.geometries) if fixed.total[start + g * _GEOMETRY_CELLS] > 0]
    scopes = [("all", present)] + [
        (str(int(separation[g])) if np.isfinite(separation[g]) else f"geometry_{g}", [g]) for g in present
    ]

    def cells(geometries, slot):
        return [start + g * _GEOMETRY_CELLS + slot for g in geometries]

    table_rows = []
    for key, geometries in scopes:
        evaluated, entangled = cells(geometries, 0), cells(geometries, 1)
        reference: dict[str, tuple[float, np.ndarray]] = {}
        for position, (label, _, _) in enumerate(CONNECTIVITY_DEFINITIONS):
            offset = 2 + 4 * position
            inside, inside_ent = cells(geometries, offset), cells(geometries, offset + 1)
            gap_out, ent_out = cells(geometries, offset + 2), cells(geometries, offset + 3)
            original = cells(geometries, 2)
            original_ent = cells(geometries, 3)
            stats = {
                "p_connected": (fixed.vector(inside), fixed.vector(evaluated), "evaluated"),
                "kappa": (fixed.vector(inside_ent), fixed.vector(inside), "evaluated"),
                "eta": (fixed.vector(entangled, minus=inside_ent), fixed.vector(evaluated, minus=inside), "evaluated"),
                # Gap pairs this definition calls disconnected, and entangled
                # connected pairs it gives up -- the latter now count toward eta.
                "explained_gap_fraction": (fixed.vector(gap_out), fixed.vector(original, minus=original_ent), "gap"),
                "entangled_cost_fraction": (fixed.vector(ent_out), fixed.vector(original_ent), "entangled_connected"),
            }
            summary = {"definition": label}
            for statistic, (top, bottom, group) in stats.items():
                value, draws, n = fixed.ratio(top, bottom)
                ci = _interval(draws)
                add("connectivity", label, statistic, group, key, value, ci, n)
                summary.update({statistic: value, f"{statistic}_ci_low": ci[0], f"{statistic}_ci_high": ci[1],
                                f"{statistic}_n": n})
                if statistic in ("kappa", "eta"):
                    if position == 0:
                        reference[statistic] = (value, draws)
                    else:
                        base_value, base_draws = reference[statistic]
                        ci = _interval(draws - base_draws)
                        add("connectivity", label, f"{statistic}_delta", "evaluated", key, value - base_value, ci, n)
                        summary.update({f"{statistic}_delta": value - base_value,
                                        f"{statistic}_delta_ci_low": ci[0], f"{statistic}_delta_ci_high": ci[1]})
            if key == "all":
                table_rows.append(summary)
    add("connectivity", "records", "count", "evaluated", "all",
        float(fixed.total[[start + g * _GEOMETRY_CELLS for g in present]].sum()), n=population.records)
    return pd.DataFrame(table_rows)


def _matched_rows(pairs: pd.DataFrame, context: dict[str, Any], *, n_boot: int, seed: int, add) -> pd.DataFrame:
    """Section matched: gap pairs against their matched controls."""
    for level in TURN_MATCH_LEVELS:
        add("matched", "match_level", "count", "gap", level, float((pairs["match_level"] == level).sum()), n=len(pairs))
    match_levels = (
        pairs.groupby(["zero_class", "match_level"]).size().rename("pairs").reset_index()
        if len(pairs)
        else pd.DataFrame(columns=["zero_class", "match_level", "pairs"])
    )
    matched = pairs.loc[pairs["match_level"] != "none"].reset_index(drop=True)
    if matched.empty:
        return match_levels
    book = _ClusterBook(matched["realization_id"].to_numpy())
    everything = np.ones(len(matched), dtype=bool)
    groups = [("gap", "control", everything)]
    if context["classifiable"]:
        for zero_name in ZERO_CLASSES:
            mask = (matched["zero_class"] == zero_name).to_numpy()
            groups.append((f"gap_{zero_name}", f"control_{zero_name}", mask))

    def column(prefix: str, name: str) -> np.ndarray:
        return matched[prefix + name].to_numpy(dtype=float)

    # One value set per turn metric, shared by both sides, so the two
    # distributions are tabulated on the same keys.
    keys: dict[str, list[int]] = {}
    for metric in TURN_METRICS:
        values = np.concatenate([column("", metric), column("control_", metric)])
        keys[metric] = sorted({int(v) for v in values[np.isfinite(values)]})

    medians: list[tuple] = []
    for gap_label, control_label, mask in groups:
        for prefix, label in (("", gap_label), ("control_", control_label)):
            for metric in TURN_METRICS:
                v = column(prefix, metric)
                finite = mask & np.isfinite(v)
                for key in keys[metric] + [TURN_BUCKET_KEY]:
                    hit = finite & ((v >= TURN_BUCKET) if key == TURN_BUCKET_KEY else (v == key))
                    book.ratio(("matched", metric, "fraction", label, key), hit, finite)
                book.ratio(("matched", metric, "mean", label, "all"), np.where(finite, v, 0.0), finite)
                add("matched", metric, "median", label, "all",
                    float(np.median(v[finite])) if finite.any() else np.nan, n=finite.sum())
                add("matched", metric, "count", label, "nan", float((mask & ~np.isfinite(v)).sum()), n=mask.sum())
            for metric, path in zip(CLASS_METRICS, PATH_PREFIXES):
                v = column(prefix, metric)
                finite = mask & np.isfinite(v)
                for code, name in enumerate(PATH_CLASSES):
                    book.ratio(("matched", metric, "fraction", label, name), finite & (v == code), finite)
                book.ratio(("matched", f"{path}_trace_only_bridge", "fraction", label, "trace_only"),
                           finite & (v == TRACE_ONLY_CLASS), finite)
                add("matched", metric, "count", label, "nan", float((mask & ~np.isfinite(v)).sum()), n=mask.sum())
                bridges = column(prefix, f"{path}_bridges")
                trace_only = column(prefix, f"{path}_bridges_trace_only")
                known = mask & np.isfinite(trace_only)
                book.ratio(("matched", f"{path}_bridges_trace_only", "fraction_positive", label, ">0"),
                           known & (trace_only > 0), known)
                both = known & np.isfinite(bridges)
                book.ratio(("matched", f"{path}_bridges_trace_only", "share_of_bridges", label, "all"),
                           np.where(both, trace_only, 0.0), np.where(both, bridges, 0.0))
                cancelled = column(prefix, f"{path}_cancelled_gates")
                known = mask & np.isfinite(cancelled)
                book.ratio(("matched", f"{path}_cancelled_gates", "fraction_positive", label, ">0"),
                           known & (cancelled > 0), known)
            for metric in MEAN_METRICS:
                v = column(prefix, metric)
                finite = mask & np.isfinite(v)
                book.ratio(("matched", metric, "mean", label, "all"), np.where(finite, v, 0.0), finite)
            for name in REFINED_FLAGS:
                v = column(prefix, name)
                finite = mask & np.isfinite(v)
                book.ratio(("matched", name, "fraction", label, "1"), finite & (v == 1.0), finite)

        # Matched differences, gap minus its own control, both finite.
        for metric in TURN_METRICS + MEAN_METRICS + REFINED_FLAGS + ("sm_trace_only_bridge", "wd_trace_only_bridge"):
            if metric.endswith("_trace_only_bridge"):
                path = metric[:2]
                gap_values = np.where(np.isfinite(column("", f"{path}_class")),
                                      (column("", f"{path}_class") == TRACE_ONLY_CLASS).astype(float), np.nan)
                control_values = np.where(np.isfinite(column("control_", f"{path}_class")),
                                          (column("control_", f"{path}_class") == TRACE_ONLY_CLASS).astype(float), np.nan)
            else:
                gap_values, control_values = column("", metric), column("control_", metric)
            both = mask & np.isfinite(gap_values) & np.isfinite(control_values)
            difference = np.where(both, gap_values - control_values, 0.0)
            book.ratio(("matched", metric, "matched_mean_difference", gap_label, "all"), difference, both)
            if metric in TURN_METRICS:
                medians.append((metric, gap_label, difference[both], book.cluster[both]))

    # P(gap | N_turn = k) inside the matched sample: half of it is gap by
    # construction, so 1/2 is the no-effect value.
    for metric in TURN_METRICS:
        g, c = column("", metric), column("control_", metric)
        for key in keys[metric] + [TURN_BUCKET_KEY]:
            if key == TURN_BUCKET_KEY:
                gap_hit, control_hit = np.isfinite(g) & (g >= TURN_BUCKET), np.isfinite(c) & (c >= TURN_BUCKET)
            else:
                gap_hit, control_hit = g == key, c == key
            book.ratio(("matched", metric, "p_gap", "matched", key), gap_hit, gap_hit.astype(float) + control_hit)

    stat_seed = seed * 7919 + _MATCHED_SEED_OFFSET
    for meta, value, ci, n in book.resolve(n_boot, stat_seed):
        add(*meta, value, ci, n)
    for metric, label, values, cluster in medians:
        if label != "gap" or values.size == 0:
            # Medians of matched differences carry a CI for the whole gap group
            # only: a weighted-quantile bootstrap per zero class would double
            # the cost for a class that is 99.96% of the pairs in practice.
            add("matched", metric, "matched_median_difference", label, "all",
                float(np.median(values)) if values.size else np.nan, n=values.size)
            continue
        draws = _weighted_quantile_draws(values, cluster, book.clusters, 0.5, n_boot, stat_seed + 1)
        add("matched", metric, "matched_median_difference", label, "all", float(np.median(values)),
            _interval(draws), n=values.size)
    return match_levels


# ---------------------------------------------------------------------------
# The selection for `dist_graph_replay.exe paths`
# ---------------------------------------------------------------------------


def _selection_frame(pairs: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Every gap pair and every matched control, once each, sorted by realization."""
    entanglement = context["entanglement"]
    gap_rows = pd.DataFrame(
        {
            "realization_id": pairs["realization_id"],
            "pair_i": pairs["pair_i"],
            "pair_j": pairs["pair_j"],
            "group": "gap",
            "geometry_id": pairs["geometry_id"],
            "embedding_id": pairs["embedding_id"],
            "separation": pairs["separation"],
            "match_level": pairs["match_level"],
            "component_size": pairs["component_size"],
            "shortest_path": pairs["shortest_path"],
            "fmn": pairs[entanglement],
        }
    )
    matched = pairs.loc[pairs["match_level"] != "none"]
    separation = context["separation"]
    control_rows = pd.DataFrame(
        {
            "realization_id": matched["realization_id"],
            "pair_i": matched["control_pair_i"],
            "pair_j": matched["control_pair_j"],
            "group": "control",
            "geometry_id": matched["control_geometry_id"],
            "embedding_id": matched["control_embedding_id"],
            "separation": separation[matched["control_geometry_id"].to_numpy(dtype=np.int64)]
            if len(matched)
            else np.empty(0),
            "match_level": matched["match_level"],
            "component_size": matched["control_component_size"],
            "shortest_path": matched["control_shortest_path"],
            "fmn": matched["control_" + entanglement],
        }
    )
    # A control matched to several gap pairs is replayed once, under the best
    # level it was matched at.
    rank = {level: position for position, level in enumerate(TURN_MATCH_LEVELS)}
    control_rows = (
        control_rows.assign(_rank=control_rows["match_level"].map(rank))
        .sort_values(["realization_id", "pair_i", "pair_j", "_rank"], kind="stable")
        .drop_duplicates(["realization_id", "pair_i", "pair_j"])
        .drop(columns="_rank")
    )
    selection = pd.concat([gap_rows, control_rows], ignore_index=True)
    group_rank = {name: position for position, name in enumerate(SELECTION_GROUPS)}
    selection = (
        selection.assign(_group=selection["group"].map(group_rank))
        .sort_values(["realization_id", "_group", "pair_i", "pair_j"], kind="stable")
        .drop(columns="_group")
        .reset_index(drop=True)
    )
    return selection[list(SELECTION_COLUMNS)]


def _write_selection(selection: pd.DataFrame, path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
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
                        _format(row.separation, integer=True),
                        row.match_level,
                        _format(row.component_size, integer=True),
                        _format(row.shortest_path, integer=True),
                        _format(row.fmn),
                    )
                )
                + "\n"
            )


def _print_turn_summary(result: dict[str, Any], records: str | Path) -> None:
    summary = result["summary"]

    def pick(section, metric, statistic, group, key="all"):
        row = summary.loc[
            (summary["section"] == section)
            & (summary["metric"] == metric)
            & (summary["statistic"] == statistic)
            & (summary["group"] == group)
            & (summary["key"] == key)
        ]
        return row.iloc[0] if len(row) else None

    def text(row, digits=4):
        if row is None or not np.isfinite(row["value"]):
            return "n/a"
        ci = (
            f" [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
            if np.isfinite(row["ci_low"])
            else ""
        )
        return f"{row['value']:.{digits}f}{ci}"

    meta = {row["metric"] + "|" + row["group"]: row["value"] for _, row in summary.loc[summary["section"] == "meta"].iterrows()}
    gap = pick("conditional", "connected", "count", "gap")
    connected = pick("conditional", "connected", "count", "connected")
    rule = "fn_positive flag" if "zero_tol|fn_positive_flag" in meta else f"fmn > {meta.get('zero_tol|explicit', np.nan):.3g}"
    print(f"Temporal-turn test: {Path(records).name}")
    print(
        f"  {int(meta.get('trajectories|all', 0)):,} trajectories, {int(meta.get('records|all', 0)):,} records; "
        f"{int(gap['value']) if gap is not None else 0:,} gap of {int(connected['value']) if connected is not None else 0:,} "
        f"connected pairs, in {int(meta.get('trajectories|gap', 0)):,} trajectories; entangled = {rule}"
    )
    print("  definition      P(C_D)    kappa [95% CI]                eta [95% CI]                  "
          "explained gap   entangled cost")
    for row in result["kappa_eta"].itertuples(index=False):
        line = (
            f"  {row.definition:14s}  {row.p_connected:.4f}    "
            f"{row.kappa:.5f} [{row.kappa_ci_low:.5f}, {row.kappa_ci_high:.5f}]  "
            f"{row.eta:.5f} [{row.eta_ci_low:.5f}, {row.eta_ci_high:.5f}]  "
            f"{row.explained_gap_fraction:.4f}          {row.entangled_cost_fraction:.2e}"
        )
        print(line)
    print(
        "  P(gap | sm class): "
        + "; ".join(f"{name} {text(pick('class', 'sm_class', 'p_gap', 'connected', name))}" for name in PATH_CLASSES)
    )
    for prefix in PATH_PREFIXES:
        name = f"{prefix}_cancelled_gates"
        print(
            f"  P(gap | {name} > 0) = {text(pick('conditional', name, 'p_gap', 'connected', '>0'))}; "
            f"= 0: {text(pick('conditional', name, 'p_gap', 'connected', '0'))}"
        )
    levels = ", ".join(
        f"{level} {int(pick('matched', 'match_level', 'count', 'gap', level)['value']):,}"
        for level in TURN_MATCH_LEVELS
        if pick("matched", "match_level", "count", "gap", level) is not None
    )
    print(f"  matched sample: {levels}")
    print(
        "  matched gap - control: sm_turns "
        f"{text(pick('matched', 'sm_turns', 'matched_mean_difference', 'gap'), 3)}, wd_turns "
        f"{text(pick('matched', 'wd_turns', 'matched_mean_difference', 'gap'), 3)}, min_turns "
        f"{text(pick('matched', 'min_turns', 'matched_mean_difference', 'gap'), 3)}"
    )


# ---------------------------------------------------------------------------
# The paths tool's output
# ---------------------------------------------------------------------------

# Columns that identify a row rather than describe its path.
_PATH_ID_COLUMNS = ("realization_id", "pair_i", "pair_j", "geometry_id", "embedding_id", "separation")
# Text columns whose values are ;-separated lists in path order.
_PATH_LIST_SUFFIXES = ("_layers", "_statuses", "_nodes", "_edges", "_modes", "_bridges_list")
_PATH_NAME_VOCABULARY = frozenset(PATH_CLASSES + CONNECTOR_STATUSES + BRIDGE_STATUSES + SELECTION_GROUPS
                                  + TURN_MATCH_LEVELS)
# Real-valued columns stay float64 even when every stored value is integral.
_PATH_REAL_NAMES = frozenset(("d", "chord"))
_PATH_REAL_SUFFIXES = ("fmn", "fmi", "_ep", "_mix", "_weight", "_bottleneck")


def read_turn_paths(path: str | Path) -> pd.DataFrame:
    """The CSV ``dist_graph_replay.exe paths`` writes, with proper dtypes.

    Identifiers become int64; a column holding only integers (counts, 0/1
    indicators) becomes int64, or the nullable ``Int64`` when it has ``nan``
    entries -- an unavailable value, never a zero; other numbers become
    float64; a column of names (path class, connector status, bridge status,
    group) becomes a categorical; and a ``;``-separated list (gate layers,
    bridge statuses, in path order) stays a string, whatever it looks like.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for name in raw.columns:
        text = raw[name].str.strip()
        missing = text.isin(["", "nan", "NaN", "NA"])
        values = text.mask(missing)
        listed = name.endswith(_PATH_LIST_SUFFIXES) or values.dropna().str.contains(";").any()
        if listed:
            out[name] = values.astype(object)
            continue
        numbers = pd.to_numeric(values, errors="coerce")
        if (numbers.notna() | missing).all() and (~missing).any():
            finite = numbers.dropna()
            integral = bool(np.all(np.isfinite(finite)) and np.all(finite == np.round(finite)))
            real = name in _PATH_REAL_NAMES or name.endswith(_PATH_REAL_SUFFIXES)
            if integral and (name in _PATH_ID_COLUMNS or not real):
                out[name] = numbers.astype("Int64") if missing.any() else numbers.astype(np.int64)
            else:
                out[name] = numbers.astype(float)
        elif (~missing).any():
            out[name] = values.astype("category")
        else:
            out[name] = numbers.astype(float)
    return pd.DataFrame(out)


def turn_path_report(paths_csv: str | Path | pd.DataFrame, *, show: bool = True) -> dict[str, pd.DataFrame]:
    """Gap against control in the paths tool's output.

    Every name column (path class, connector status, ...) is tabulated by
    group, with counts and within-group fractions; every ``;``-list of names
    (bridge statuses along a path) is tabulated per element, so a bridge is
    counted once wherever it sits; and every numeric column gets its per-group
    count, NaN count, mean and fraction positive. Returns ``{"frame",
    "categories", "lists", "numeric"}``, the last three long-format.
    """
    frame = paths_csv.copy() if isinstance(paths_csv, pd.DataFrame) else read_turn_paths(paths_csv)
    if "group" not in frame.columns:
        raise ValueError("The paths CSV has no `group` column; it is not dist_graph_replay.exe paths output.")
    group = frame["group"].astype(str)
    categories, lists, numeric = [], [], []

    def tabulate(labels: pd.Series, values: pd.Series, column: str, into: list) -> None:
        # Arrays, not Series: an exploded list repeats its index labels, and
        # crosstab would align on them.
        table = pd.crosstab(np.asarray(labels, dtype=object), np.asarray(values, dtype=object))
        totals = table.sum(axis=1)
        for name in table.index:
            for value in table.columns:
                count = int(table.loc[name, value])
                into.append({"column": column, "group": name, "value": value, "count": count,
                             "fraction": count / totals[name] if totals[name] else np.nan})

    for column in frame.columns:
        if column == "group" or column in _PATH_ID_COLUMNS:
            continue
        series = frame[column]
        if series.dtype == object:
            tokens = series.dropna().astype(str).str.split(";").explode()
            tokens = tokens[tokens != ""]
            if len(tokens) and tokens.isin(_PATH_NAME_VOCABULARY).all():
                tabulate(group.loc[tokens.index], tokens, column, lists)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            tabulate(group, series.astype(object).where(series.notna(), "nan"), column, categories)
        else:
            values = pd.to_numeric(series, errors="coerce").astype(float)
            for name in sorted(group.unique()):
                selected = values[group == name]
                finite = selected.dropna()
                numeric.append({"column": column, "group": name, "count": int(finite.size),
                                "nan_count": int(selected.isna().sum()),
                                "mean": float(finite.mean()) if finite.size else np.nan,
                                "fraction_positive": float((finite > 0).mean()) if finite.size else np.nan})
    report = {
        "frame": frame,
        "categories": pd.DataFrame(categories, columns=["column", "group", "value", "count", "fraction"]),
        "lists": pd.DataFrame(lists, columns=["column", "group", "value", "count", "fraction"]),
        "numeric": pd.DataFrame(numeric, columns=["column", "group", "count", "nan_count", "mean", "fraction_positive"]),
    }
    if show:
        for key in ("categories", "lists"):
            table = report[key]
            for column, rows in table.groupby("column", sort=False):
                cells = rows.pivot(index="value", columns="group", values="fraction").fillna(0.0)
                print(f"{column} ({'per path' if key == 'categories' else 'per element'}), fraction by group:")
                print(cells.to_string(float_format=lambda x: f"{x:.4f}"))
        numbers = report["numeric"]
        if len(numbers):
            print(numbers.pivot(index="column", columns="group", values="mean").to_string(float_format=lambda x: f"{x:.4g}"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_analysis.turns",
        description="Temporal-turn test on a turn-augmented dist_scaling record binary.",
    )
    parser.add_argument("records", help="a <stem>_turns_records.bin (dist_graph_replay.exe augment)")
    parser.add_argument("--zero-tol", type=float, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-records", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--selection-output", default=None)
    args = parser.parse_args(argv)
    result = temporal_turn_test(
        args.records,
        zero_tol=args.zero_tol,
        n_boot=args.n_boot,
        seed=args.seed,
        chunk_records=args.chunk_records,
        output=args.output,
        selection_output=args.selection_output,
    )
    print(f"wrote {result['path']} and {result['selection_path']} ({len(result['selection'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

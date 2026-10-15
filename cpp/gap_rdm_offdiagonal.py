#!/usr/bin/env python3
"""Stream MIPT pair-gap rho3/rho4 companions and inspect silent pair marginals.

The files written by ``dist_scaling.exe`` are raw, fixed-width fermionic RDM
companions.  This program answers two different questions which should not be
conflated:

1. Does a two-mode marginal classified as fN-zero still have off-diagonal
   entries?  In a parity-preserving two-mode RDM the only physical entries are
   F = -rho[11,00] and G = rho[01,10] (using the program's bit ordering), and
   the cancellation-free closed form for fN is used here.
2. Can the containing three- or four-mode RDM have off-diagonal coherence even
   when that pair marginal has fN below the chosen threshold?

The scan is chunked and does not load the 1--3 GB inputs into memory.  With an
optional ``connected_zero_pair_summary.csv``, only the actual connected-zero
anchor pairs are counted and repeated occurrences are deduplicated.  Without
that CSV every pair marginal in every stored RDM is examined; this is enough
to produce examples, but it cannot tell which pair was the percolation anchor.

Examples
--------
    python3 gap_rdm_offdiagonal.py \
      rppu_..._connected_zero_thirds_rho3.bin \
      rppu_..._connected_zero_thirds_rho4.bin \
      --pair-summary rppu_..._connected_zero_pair_summary.csv \
      --output-prefix rppu_gap_offdiag

    python3 gap_rdm_offdiagonal.py \
      rppu_..._connected_zero_thirds_ppt_ccnr.bin \
      --output-prefix ppt_ccnr_check

Outputs are ``<prefix>_summary.csv``, ``<prefix>_examples.txt``, and an NPZ
containing the full-precision example matrices.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
import time
from typing import Iterable

import numpy as np


MAGIC = b"MIPTGAP3"
PREAMBLE_BYTES = 24


def parse_key_value_header(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


@dataclass(frozen=True)
class CompanionInfo:
    path: Path
    parties: int
    dimension: int
    record_bytes: int
    declared_record_bytes: int
    data_offset: int
    record_count: int
    trailing_bytes: int
    header: dict[str, str]
    header_text: str


def infer_parties(path: Path, requested: str, payload_bytes: int) -> int:
    if requested != "auto":
        return int(requested)
    lower = path.name.lower()
    if "rho4" in lower:
        return 4
    if "rho3" in lower or "ppt_ccnr" in lower:
        return 3

    sizes = {3: 16 + 128 * 8, 4: 16 + 512 * 8}
    fits = [parties for parties, size in sizes.items() if payload_bytes % size == 0]
    if len(fits) == 1:
        return fits[0]
    raise ValueError(
        f"Cannot infer whether {path} contains rho3 or rho4 records. "
        "Pass --parties 3 or --parties 4."
    )


def read_companion_info(path: str | Path, requested_parties: str = "auto") -> CompanionInfo:
    path = Path(path)
    with path.open("rb") as handle:
        preamble = handle.read(PREAMBLE_BYTES)
        if len(preamble) != PREAMBLE_BYTES:
            raise ValueError(f"{path} is shorter than the 24-byte companion preamble.")
        magic, version, declared_record_bytes, header_bytes, _reserved = struct.unpack(
            "<8sIIII", preamble
        )
        if magic != MAGIC:
            raise ValueError(f"{path} has magic {magic!r}, not {MAGIC!r}.")
        if version != 1:
            raise ValueError(f"{path} has unsupported companion version {version}.")
        header_raw = handle.read(header_bytes)
        if len(header_raw) != header_bytes:
            raise ValueError(f"{path} ends inside its text header.")
    header_text = header_raw.decode("utf-8")
    header = parse_key_value_header(header_text)
    data_offset = PREAMBLE_BYTES + header_bytes
    payload_bytes = path.stat().st_size - data_offset
    if payload_bytes < 0:
        raise ValueError(f"{path} has an invalid header length.")
    parties = infer_parties(path, requested_parties, payload_bytes)
    doubles = 2 * (1 << parties) ** 2
    record_bytes = 16 + doubles * 8
    count, trailing = divmod(payload_bytes, record_bytes)
    return CompanionInfo(
        path=path,
        parties=parties,
        dimension=1 << parties,
        record_bytes=record_bytes,
        declared_record_bytes=declared_record_bytes,
        data_offset=data_offset,
        record_count=count,
        trailing_bytes=trailing,
        header=header,
        header_text=header_text,
    )


def companion_dtype(parties: int) -> np.dtype:
    doubles = 2 * (1 << parties) ** 2
    # rho4 companions currently reuse the rho3 preamble; the actual record is
    # nevertheless unambiguous: 16 header bytes followed by 512 float64s.
    return np.dtype(
        {
            "names": ("realization_id", "record_id", "sites", "rho"),
            "formats": ("<u4", "<u4", ("u1", (parties,)), ("<f8", (doubles,))),
            "offsets": (0, 4, 8, 16),
            "itemsize": 16 + doubles * 8,
        }
    )


def iter_chunks(info: CompanionInfo, chunk_records: int) -> Iterable[np.ndarray]:
    dtype = companion_dtype(info.parties)
    with info.path.open("rb") as handle:
        handle.seek(info.data_offset)
        remaining = info.record_count
        while remaining:
            count = min(chunk_records, remaining)
            block = np.fromfile(handle, dtype=dtype, count=count)
            if block.size != count:
                raise IOError(f"Unexpected end of {info.path} while reading its payload.")
            yield block
            remaining -= count


def normalized_hermitian(raw: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    shaped = raw.reshape(raw.shape[0], dimension, dimension, 2)
    matrix = shaped[..., 0] + 1j * shaped[..., 1]
    trace = np.trace(matrix, axis1=1, axis2=2).real
    if np.any(~np.isfinite(trace)) or np.any(trace <= 0.0):
        raise ValueError("A companion RDM has a non-positive or non-finite trace.")
    matrix = 0.5 * (matrix + matrix.swapaxes(1, 2).conj())
    matrix /= trace[:, None, None]
    return matrix, trace


def reordered_position(mode: int, retained_mask: int, modes: int) -> int:
    retained = [candidate for candidate in range(modes) if (retained_mask >> candidate) & 1]
    environment = [candidate for candidate in range(modes) if not ((retained_mask >> candidate) & 1)]
    return (retained + environment).index(mode)


def fermion_reorder_sign(occupation: int, retained_mask: int, modes: int) -> float:
    seen = 0
    parity = 0
    for mode in range(modes):
        if not ((occupation >> mode) & 1):
            continue
        position = reordered_position(mode, retained_mask, modes)
        lower_or_equal = (1 << (position + 1)) - 1
        parity ^= ((seen & ~lower_or_equal).bit_count() & 1)
        seen |= 1 << position
    return -1.0 if parity else 1.0


def embed_local_bits(retained: int, environment: int, retained_mask: int, modes: int) -> int:
    full = 0
    retained_bit = 0
    environment_bit = 0
    for mode in range(modes):
        if (retained_mask >> mode) & 1:
            bit = (retained >> retained_bit) & 1
            retained_bit += 1
        else:
            bit = (environment >> environment_bit) & 1
            environment_bit += 1
        full |= bit << mode
    return full


def partial_trace_plan(modes: int, retained_mask: int) -> tuple[np.ndarray, np.ndarray]:
    retained_count = retained_mask.bit_count()
    local_dimension = 1 << retained_count
    environment_dimension = 1 << (modes - retained_count)
    indices = np.empty((local_dimension, environment_dimension), dtype=np.intp)
    signs = np.empty((local_dimension, environment_dimension), dtype=np.float64)
    for local in range(local_dimension):
        for environment in range(environment_dimension):
            full = embed_local_bits(local, environment, retained_mask, modes)
            indices[local, environment] = full
            signs[local, environment] = fermion_reorder_sign(full, retained_mask, modes)
    return indices, signs


def fermionic_partial_trace(
    rho: np.ndarray, plan: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    indices, signs = plan
    local_dimension, environment_dimension = indices.shape
    out = np.zeros((rho.shape[0], local_dimension, local_dimension), dtype=np.complex128)
    for environment in range(environment_dimension):
        index = indices[:, environment]
        sign = signs[:, environment]
        out += rho[:, index[:, None], index[None, :]] * sign[:, None] * sign[None, :]
    return out


def pair_metrics(pair: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.diagonal(pair, axis1=1, axis2=2).real
    parity_even = diagonal[:, 0] + diagonal[:, 3]
    parity_odd = diagonal[:, 1] + diagonal[:, 2]
    g = pair[:, 2, 1]
    f = -pair[:, 3, 0]
    g2 = np.square(np.abs(g))
    f2 = np.square(np.abs(f))

    # Cancellation-free exact expression used by dist_scaling.exe.
    denom_g = np.sqrt(np.square(parity_even) + 4.0 * g2) + parity_even
    denom_f = np.sqrt(np.square(parity_odd) + 4.0 * f2) + parity_odd
    term_g = np.divide(2.0 * g2, denom_g, out=np.zeros_like(g2), where=denom_g > 0.0)
    term_f = np.divide(2.0 * f2, denom_f, out=np.zeros_like(f2), where=denom_f > 0.0)
    fn = term_g + term_f

    offdiag = np.abs(pair).copy()
    diagonal_index = np.arange(4)
    offdiag[:, diagonal_index, diagonal_index] = 0.0
    pair_offdiag_max = offdiag.max(axis=(1, 2))

    parity = np.array([0, 1, 1, 0], dtype=np.int8)
    same_parity = parity[:, None] == parity[None, :]
    allowed_mask = same_parity & ~np.eye(4, dtype=bool)
    leakage_mask = ~same_parity
    allowed_offdiag_max = np.abs(pair[:, allowed_mask]).max(axis=1)
    parity_leakage_max = np.abs(pair[:, leakage_mask]).max(axis=1)

    return {
        "fn": fn,
        "f": f,
        "g": g,
        "f2": f2,
        "g2": g2,
        "pair_offdiag_max": pair_offdiag_max,
        "pair_allowed_offdiag_max": allowed_offdiag_max,
        "pair_parity_leakage_max": parity_leakage_max,
    }


def full_offdiagonal_metrics(rho: np.ndarray, parties: int) -> dict[str, np.ndarray]:
    dimension = 1 << parties
    absolute = np.abs(rho).copy()
    diagonal = np.arange(dimension)
    absolute[:, diagonal, diagonal] = 0.0
    parity = np.array([index.bit_count() & 1 for index in range(dimension)], dtype=np.int8)
    same_parity = parity[:, None] == parity[None, :]
    allowed_mask = same_parity & ~np.eye(dimension, dtype=bool)
    leakage_mask = ~same_parity
    return {
        "full_offdiag_max": absolute.max(axis=(1, 2)),
        "full_allowed_offdiag_max": np.abs(rho[:, allowed_mask]).max(axis=1),
        "full_parity_leakage_max": np.abs(rho[:, leakage_mask]).max(axis=1),
    }


def encode_pair_keys(realization: np.ndarray, site_i: np.ndarray, site_j: np.ndarray) -> np.ndarray:
    low = np.minimum(site_i, site_j).astype(np.uint64)
    high = np.maximum(site_i, site_j).astype(np.uint64)
    return (realization.astype(np.uint64) << np.uint64(16)) | (low << np.uint64(8)) | high


def load_anchor_keys(path: str | Path) -> np.ndarray:
    keys: list[int] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"realization_id", "anchor_role", "pair_i", "pair_j"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is not a connected-zero pair-summary CSV.")
        for row in reader:
            if row["anchor_role"] != "anchor":
                continue
            realization = int(row["realization_id"])
            i, j = sorted((int(row["pair_i"]), int(row["pair_j"])))
            keys.append((realization << 16) | (i << 8) | j)
    return np.unique(np.asarray(keys, dtype=np.uint64))


def membership(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    if sorted_keys.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    positions = np.searchsorted(sorted_keys, keys)
    valid = positions < sorted_keys.size
    result = np.zeros(keys.shape, dtype=bool)
    result[valid] = sorted_keys[positions[valid]] == keys[valid]
    return result


@dataclass
class Example:
    score: float
    kind: str
    source: str
    realization: int
    record_id: int
    sites: np.ndarray
    pair_positions: tuple[int, int]
    pair_sites: tuple[int, int]
    fn: float
    f: complex
    g: complex
    pair_offdiag_max: float
    pair_allowed_offdiag_max: float
    pair_parity_leakage_max: float
    full_offdiag_max: float
    full_allowed_offdiag_max: float
    full_parity_leakage_max: float
    pair_rho: np.ndarray
    full_rho: np.ndarray


def maybe_replace_example(current: Example | None, candidate: Example) -> Example:
    return candidate if current is None or candidate.score > current.score else current


def make_example(
    kind: str,
    score: float,
    info: CompanionInfo,
    block: np.ndarray,
    rho: np.ndarray,
    pair: np.ndarray,
    metrics: dict[str, np.ndarray],
    full_metrics: dict[str, np.ndarray],
    index: int,
    pair_positions: tuple[int, int],
) -> Example:
    sites = block["sites"][index].astype(int).copy()
    pair_sites = tuple(sorted((int(sites[pair_positions[0]]), int(sites[pair_positions[1]]))))
    return Example(
        score=float(score),
        kind=kind,
        source=str(info.path),
        realization=int(block["realization_id"][index]),
        record_id=int(block["record_id"][index]),
        sites=sites,
        pair_positions=pair_positions,
        pair_sites=pair_sites,
        fn=float(metrics["fn"][index]),
        f=complex(metrics["f"][index]),
        g=complex(metrics["g"][index]),
        pair_offdiag_max=float(metrics["pair_offdiag_max"][index]),
        pair_allowed_offdiag_max=float(metrics["pair_allowed_offdiag_max"][index]),
        pair_parity_leakage_max=float(metrics["pair_parity_leakage_max"][index]),
        full_offdiag_max=float(full_metrics["full_offdiag_max"][index]),
        full_allowed_offdiag_max=float(full_metrics["full_allowed_offdiag_max"][index]),
        full_parity_leakage_max=float(full_metrics["full_parity_leakage_max"][index]),
        pair_rho=pair[index].copy(),
        full_rho=rho[index].copy(),
    )


def scan_file(
    info: CompanionInfo,
    *,
    anchor_keys: np.ndarray | None,
    fn_zero_tol: float,
    pair_offdiag_tol: float,
    full_offdiag_tol: float,
    chunk_records: int,
    progress_every: int,
) -> tuple[dict[str, object], list[Example]]:
    pair_positions = list(itertools.combinations(range(info.parties), 2))
    plans = {
        positions: partial_trace_plan(
            info.parties, (1 << positions[0]) | (1 << positions[1])
        )
        for positions in pair_positions
    }
    counters: dict[str, int] = {
        "records": 0,
        "pair_marginal_occurrences": 0,
        "pair_marginals_tested": 0,
        "fn_zero": 0,
        "fn_zero_pair_any_offdiag_above_tol": 0,
        "fn_zero_pair_allowed_offdiag_above_tol": 0,
        "fn_zero_full_any_offdiag_above_tol": 0,
        "fn_zero_full_allowed_offdiag_above_tol": 0,
        "fn_zero_pair_parity_leakage_above_tol": 0,
        "fn_zero_full_parity_leakage_above_tol": 0,
    }
    maxima = {
        "max_pair_offdiag_among_fn_zero": 0.0,
        "max_pair_allowed_offdiag_among_fn_zero": 0.0,
        "max_pair_parity_leakage_among_fn_zero": 0.0,
        "max_full_offdiag_among_fn_zero": 0.0,
        "max_full_allowed_offdiag_among_fn_zero": 0.0,
        "max_full_parity_leakage_among_fn_zero": 0.0,
    }
    best_pair: Example | None = None
    best_full: Example | None = None
    seen: set[int] | None = set() if anchor_keys is not None else None
    started = time.monotonic()
    next_progress = progress_every

    if info.record_count == 0:
        summary: dict[str, object] = {
            "file": str(info.path),
            "parties": info.parties,
            "records": 0,
            "selection": "connected_zero_anchors" if anchor_keys is not None else "all_pair_occurrences",
            **{key: value for key, value in counters.items() if key != "records"},
            **maxima,
            "fn_zero_tol": fn_zero_tol,
            "pair_offdiag_tol": pair_offdiag_tol,
            "full_offdiag_tol": full_offdiag_tol,
            "trailing_bytes": info.trailing_bytes,
        }
        return summary, []

    for block in iter_chunks(info, chunk_records):
        rho, _trace = normalized_hermitian(block["rho"], info.dimension)
        full_metrics = full_offdiagonal_metrics(rho, info.parties)
        counters["records"] += block.size

        for positions in pair_positions:
            pair = fermionic_partial_trace(rho, plans[positions])
            metrics = pair_metrics(pair)
            i = block["sites"][:, positions[0]]
            j = block["sites"][:, positions[1]]
            keys = encode_pair_keys(block["realization_id"], i, j)
            selected = np.ones(block.size, dtype=bool)
            counters["pair_marginal_occurrences"] += block.size

            if anchor_keys is not None:
                selected &= membership(anchor_keys, keys)
                if seen is not None and np.any(selected):
                    for index in np.flatnonzero(selected):
                        key = int(keys[index])
                        if key in seen:
                            selected[index] = False
                        else:
                            seen.add(key)

            counters["pair_marginals_tested"] += int(selected.sum())
            zero = selected & (metrics["fn"] <= fn_zero_tol)
            zero_indices = np.flatnonzero(zero)
            if zero_indices.size == 0:
                continue

            counters["fn_zero"] += int(zero_indices.size)
            pair_any_above = zero & (metrics["pair_offdiag_max"] > pair_offdiag_tol)
            pair_allowed_above = zero & (
                metrics["pair_allowed_offdiag_max"] > pair_offdiag_tol
            )
            full_any_above = zero & (full_metrics["full_offdiag_max"] > full_offdiag_tol)
            full_allowed_above = zero & (
                full_metrics["full_allowed_offdiag_max"] > full_offdiag_tol
            )
            pair_leak = zero & (metrics["pair_parity_leakage_max"] > pair_offdiag_tol)
            full_leak = zero & (full_metrics["full_parity_leakage_max"] > full_offdiag_tol)
            counters["fn_zero_pair_any_offdiag_above_tol"] += int(pair_any_above.sum())
            counters["fn_zero_pair_allowed_offdiag_above_tol"] += int(
                pair_allowed_above.sum()
            )
            counters["fn_zero_full_any_offdiag_above_tol"] += int(full_any_above.sum())
            counters["fn_zero_full_allowed_offdiag_above_tol"] += int(
                full_allowed_above.sum()
            )
            counters["fn_zero_pair_parity_leakage_above_tol"] += int(pair_leak.sum())
            counters["fn_zero_full_parity_leakage_above_tol"] += int(full_leak.sum())

            for name, values in (
                ("max_pair_offdiag_among_fn_zero", metrics["pair_offdiag_max"]),
                ("max_pair_allowed_offdiag_among_fn_zero", metrics["pair_allowed_offdiag_max"]),
                ("max_pair_parity_leakage_among_fn_zero", metrics["pair_parity_leakage_max"]),
                ("max_full_offdiag_among_fn_zero", full_metrics["full_offdiag_max"]),
                ("max_full_allowed_offdiag_among_fn_zero", full_metrics["full_allowed_offdiag_max"]),
                ("max_full_parity_leakage_among_fn_zero", full_metrics["full_parity_leakage_max"]),
            ):
                maxima[name] = max(maxima[name], float(values[zero].max()))

            pair_index = zero_indices[
                int(np.argmax(metrics["pair_allowed_offdiag_max"][zero_indices]))
            ]
            pair_candidate = make_example(
                "largest_pair_marginal_offdiagonal",
                float(metrics["pair_allowed_offdiag_max"][pair_index]),
                info,
                block,
                rho,
                pair,
                metrics,
                full_metrics,
                int(pair_index),
                positions,
            )
            best_pair = maybe_replace_example(best_pair, pair_candidate)

            full_index = zero_indices[
                int(np.argmax(full_metrics["full_allowed_offdiag_max"][zero_indices]))
            ]
            full_candidate = make_example(
                "largest_containing_rdm_offdiagonal",
                float(full_metrics["full_allowed_offdiag_max"][full_index]),
                info,
                block,
                rho,
                pair,
                metrics,
                full_metrics,
                int(full_index),
                positions,
            )
            best_full = maybe_replace_example(best_full, full_candidate)

        if progress_every > 0 and counters["records"] >= next_progress:
            elapsed = max(time.monotonic() - started, 1.0e-9)
            print(
                f"  {info.path.name}: {counters['records']:,}/{info.record_count:,} records "
                f"({counters['records'] / elapsed:,.0f}/s), "
                f"{counters['fn_zero']:,} fN-zero pairs",
                file=sys.stderr,
            )
            while next_progress <= counters["records"]:
                next_progress += progress_every

    summary = {
        "file": str(info.path),
        "parties": info.parties,
        "selection": "connected_zero_anchors_deduplicated" if anchor_keys is not None else "all_pair_occurrences",
        **counters,
        **maxima,
        "fn_zero_tol": fn_zero_tol,
        "pair_offdiag_tol": pair_offdiag_tol,
        "full_offdiag_tol": full_offdiag_tol,
        "trailing_bytes": info.trailing_bytes,
    }
    examples = [example for example in (best_pair, best_full) if example is not None]
    return summary, examples


def basis_label(index: int, parties: int) -> str:
    return "".join(str((index >> mode) & 1) for mode in range(parties))


def format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        matrix,
        precision=8,
        suppress_small=False,
        max_line_width=240,
        formatter={"complex_kind": lambda value: f"{value.real:+.8e}{value.imag:+.8e}j"},
    )


def strongest_offdiagonals(matrix: np.ndarray, parties: int, limit: int = 24) -> list[str]:
    entries: list[tuple[float, int, int, complex]] = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == col:
                continue
            value = complex(matrix[row, col])
            entries.append((abs(value), row, col, value))
    entries.sort(reverse=True, key=lambda item: item[0])
    lines = []
    for magnitude, row, col, value in entries[:limit]:
        if magnitude == 0.0:
            break
        lines.append(
            f"  |{basis_label(row, parties)}><{basis_label(col, parties)}|  "
            f"{value.real:+.12e}{value.imag:+.12e}j  abs={magnitude:.12e}"
        )
    return lines


def write_outputs(
    prefix: Path, summaries: list[dict[str, object]], examples: list[Example]
) -> tuple[Path, Path, Path]:
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    examples_path = prefix.with_name(prefix.name + "_examples.txt")
    npz_path = prefix.with_name(prefix.name + "_examples.npz")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with examples_path.open("w") as handle:
        handle.write(
            "Basis convention: local mode 0 is the least-significant bit; labels are written "
            "in ascending local/site order.\n\n"
        )
        if not examples:
            handle.write("No fN-zero pair marginal was found under the requested selection.\n")
        for number, example in enumerate(examples, start=1):
            handle.write(f"EXAMPLE {number}: {example.kind}\n")
            handle.write(f"source={example.source}\n")
            handle.write(
                f"realization={example.realization} record_id={example.record_id} "
                f"sites={example.sites.tolist()} pair_positions={example.pair_positions} "
                f"pair_sites={example.pair_sites}\n"
            )
            handle.write(
                f"fN={example.fn:.17g} F={example.f.real:+.17g}{example.f.imag:+.17g}j "
                f"G={example.g.real:+.17g}{example.g.imag:+.17g}j\n"
            )
            handle.write(
                f"pair_offdiag_max={example.pair_offdiag_max:.17g} "
                f"pair_allowed_offdiag_max={example.pair_allowed_offdiag_max:.17g} "
                f"pair_parity_leakage_max={example.pair_parity_leakage_max:.17g}\n"
            )
            handle.write(
                f"full_offdiag_max={example.full_offdiag_max:.17g} "
                f"full_allowed_offdiag_max={example.full_allowed_offdiag_max:.17g} "
                f"full_parity_leakage_max={example.full_parity_leakage_max:.17g}\n"
            )
            handle.write("\nTwo-mode marginal in basis |00>, |10>, |01>, |11>:\n")
            handle.write(format_matrix(example.pair_rho) + "\n")
            handle.write("\nStrongest off-diagonal entries of the containing RDM:\n")
            handle.write("\n".join(strongest_offdiagonals(example.full_rho, len(example.sites))))
            handle.write("\n\nFull containing RDM:\n")
            handle.write(format_matrix(example.full_rho) + "\n\n" + "=" * 100 + "\n\n")

    arrays: dict[str, np.ndarray] = {}
    for index, example in enumerate(examples):
        arrays[f"example_{index}_full_rho"] = example.full_rho
        arrays[f"example_{index}_pair_rho"] = example.pair_rho
        arrays[f"example_{index}_sites"] = example.sites
        arrays[f"example_{index}_pair_sites"] = np.asarray(example.pair_sites)
        arrays[f"example_{index}_metadata"] = np.asarray(
            [example.realization, example.record_id, example.fn, example.score], dtype=float
        )
    np.savez_compressed(npz_path, **arrays)
    return summary_path, examples_path, npz_path


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="rho3.bin, rho4.bin, or ppt_ccnr.bin companions")
    parser.add_argument(
        "--parties",
        choices=("auto", "3", "4"),
        default="auto",
        help="infer from the filename by default; override for an ambiguously named file",
    )
    parser.add_argument(
        "--pair-summary",
        default=None,
        help=(
            "optional connected_zero_pair_summary.csv; restricts the scan to actual anchor "
            "pairs and deduplicates them"
        ),
    )
    parser.add_argument("--fn-zero-tol", type=positive_float, default=1.0e-12)
    parser.add_argument(
        "--pair-offdiag-tol",
        type=positive_float,
        default=1.0e-14,
        help="FP64 amplitude floor corresponding to the run's 1e-28 squared-channel floor",
    )
    parser.add_argument("--full-offdiag-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--chunk-records", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--output-prefix", default="gap_rdm_offdiagonal")
    arguments = parser.parse_args(argv)
    if arguments.chunk_records <= 0:
        parser.error("--chunk-records must be positive")

    anchor_keys = None
    if arguments.pair_summary:
        anchor_keys = load_anchor_keys(arguments.pair_summary)
        print(f"Loaded {anchor_keys.size:,} connected-zero anchor keys.", file=sys.stderr)

    summaries: list[dict[str, object]] = []
    examples: list[Example] = []
    for filename in arguments.files:
        info = read_companion_info(filename, arguments.parties)
        print(
            f"{info.path}: parties={info.parties}, records={info.record_count:,}, "
            f"payload_record_bytes={info.record_bytes}",
            file=sys.stderr,
        )
        if info.declared_record_bytes != info.record_bytes:
            print(
                f"  NOTE: preamble declares {info.declared_record_bytes} bytes; using the "
                f"known rho{info.parties} payload width {info.record_bytes}.",
                file=sys.stderr,
            )
        if info.trailing_bytes:
            print(
                f"  WARNING: ignoring {info.trailing_bytes} trailing byte(s) that do not form "
                "a complete record.",
                file=sys.stderr,
            )
        summary, file_examples = scan_file(
            info,
            anchor_keys=anchor_keys,
            fn_zero_tol=arguments.fn_zero_tol,
            pair_offdiag_tol=arguments.pair_offdiag_tol,
            full_offdiag_tol=arguments.full_offdiag_tol,
            chunk_records=arguments.chunk_records,
            progress_every=arguments.progress_every,
        )
        summaries.append(summary)
        examples.extend(file_examples)

    outputs = write_outputs(Path(arguments.output_prefix), summaries, examples)
    for summary in summaries:
        print(
            f"{summary['file']}: tested={int(summary['pair_marginals_tested']):,}, "
            f"fN-zero={int(summary['fn_zero']):,}, "
            f"pair-physical-offdiag-above-floor="
            f"{int(summary['fn_zero_pair_allowed_offdiag_above_tol']):,}, "
            f"containing-RDM-physical-offdiag="
            f"{int(summary['fn_zero_full_allowed_offdiag_above_tol']):,}"
        )
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

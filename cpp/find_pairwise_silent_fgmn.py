#!/usr/bin/env python3
"""Find three-mode RDMs with positive fGMN and pairwise-zero fermionic negativity.

This joins the per-third outcome CSV written by ``dist_scaling.exe`` to its
``*_connected_zero_thirds_rho3.bin`` companion.  A qualifying record has

    fGMN > --fgmn-min

and all three two-mode marginal negativities (AB, AC, BC) no larger than
``--pair-zero-tol``.  The binary is streamed; it is never loaded in full.

Important terminology: this tests all three *two-mode marginals*.  It does not
ask for zero negativity on the three 1|23 bipartitions.  Positive fGMN already
requires entanglement across every 1|23 cut.

Example
-------
python3 find_pairwise_silent_fgmn.py \
  rppu_L20_T40_p0p335_r25k_gap_v2_connected_zero_thirds.csv \
  rppu_L20_T40_p0p335_r25k_gap_v2_connected_zero_thirds_rho3.bin \
  --pair-zero-tol 1e-12 --fgmn-min 1e-10 --examples 3 \
  --output-prefix rppu_pairwise_silent_fgmn

Run again with ``--pair-zero-tol 0`` to demand literal zero values in the CSV.
The report always tabulates several tighter thresholds as a numerical check.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterable

import numpy as np


MAGIC = b"MIPTGAP3"
PREAMBLE_BYTES = 24
RHO3_RECORD_BYTES = 16 + 128 * 8


@dataclass(frozen=True)
class Rho3Info:
    path: Path
    version: int
    declared_record_bytes: int
    header_bytes: int
    data_offset: int
    record_count: int
    trailing_bytes: int
    header_text: str


@dataclass
class Candidate:
    realization_id: int
    triple_id: int
    sites: tuple[int, int, int]
    fgmn: float
    fgmn_raw: float
    fgmn_method: str
    fgmn_status: str
    pair_values: dict[tuple[int, int], float]
    pair_zero_tol_recorded: float
    fgmn_positive_tol_recorded: float
    cut_values: dict[int, float]
    min_cut: float
    anchor_pairs: set[tuple[int, int]]

    @property
    def key(self) -> tuple[int, int]:
        return self.realization_id, self.triple_id

    @property
    def max_pair_fn(self) -> float:
        return max(self.pair_values.values())


def finite_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse {name}={text!r} as a float.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected finite {name}, got {text!r}.")
    return value


def read_rho3_info(path: str | Path) -> Rho3Info:
    path = Path(path)
    with path.open("rb") as handle:
        preamble = handle.read(PREAMBLE_BYTES)
        if len(preamble) != PREAMBLE_BYTES:
            raise ValueError(f"{path} is shorter than its 24-byte preamble.")
        magic, version, record_bytes, header_bytes, _reserved = struct.unpack(
            "<8sIIII", preamble
        )
        if magic != MAGIC:
            raise ValueError(f"{path} has magic {magic!r}, not {MAGIC!r}.")
        if version != 1:
            raise ValueError(f"{path} has unsupported version {version}.")
        header_raw = handle.read(header_bytes)
        if len(header_raw) != header_bytes:
            raise ValueError(f"{path} ends inside its text header.")
    header_text = header_raw.decode("utf-8")
    data_offset = PREAMBLE_BYTES + header_bytes
    payload = path.stat().st_size - data_offset
    if payload < 0:
        raise ValueError(f"{path} has an invalid header length.")
    count, trailing = divmod(payload, RHO3_RECORD_BYTES)
    if record_bytes != RHO3_RECORD_BYTES:
        raise ValueError(
            f"{path} declares {record_bytes} bytes per record; expected "
            f"{RHO3_RECORD_BYTES}."
        )
    return Rho3Info(
        path=path,
        version=version,
        declared_record_bytes=record_bytes,
        header_bytes=header_bytes,
        data_offset=data_offset,
        record_count=count,
        trailing_bytes=trailing,
        header_text=header_text,
    )


def rho3_dtype() -> np.dtype:
    return np.dtype(
        {
            "names": ("realization_id", "triple_id", "sites", "rho"),
            "formats": ("<u4", "<u4", ("u1", (3,)), ("<f8", (128,))),
            "offsets": (0, 4, 8, 16),
            "itemsize": RHO3_RECORD_BYTES,
        }
    )


def iter_rho3_chunks(info: Rho3Info, chunk_records: int) -> Iterable[np.ndarray]:
    dtype = rho3_dtype()
    with info.path.open("rb") as handle:
        handle.seek(info.data_offset)
        remaining = info.record_count
        while remaining:
            count = min(chunk_records, remaining)
            block = np.fromfile(handle, dtype=dtype, count=count)
            if block.size != count:
                raise IOError(f"Unexpected end of {info.path}.")
            yield block
            remaining -= count


def embed_local_bits(retained: int, environment: int, mask: int, modes: int) -> int:
    full = 0
    rbit = 0
    ebit = 0
    for mode in range(modes):
        if (mask >> mode) & 1:
            bit = (retained >> rbit) & 1
            rbit += 1
        else:
            bit = (environment >> ebit) & 1
            ebit += 1
        full |= bit << mode
    return full


def fermion_reorder_sign(occupation: int, mask: int, modes: int) -> float:
    retained = [mode for mode in range(modes) if (mask >> mode) & 1]
    environment = [mode for mode in range(modes) if not ((mask >> mode) & 1)]
    order = retained + environment
    new_position = {mode: position for position, mode in enumerate(order)}
    occupied = [mode for mode in range(modes) if (occupation >> mode) & 1]
    inversions = 0
    for left in range(len(occupied)):
        for right in range(left + 1, len(occupied)):
            inversions += new_position[occupied[left]] > new_position[occupied[right]]
    return -1.0 if inversions & 1 else 1.0


def partial_trace_plan(mask: int, modes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    local_dimension = 1 << mask.bit_count()
    environment_dimension = 1 << (modes - mask.bit_count())
    indices = np.empty((local_dimension, environment_dimension), dtype=np.intp)
    signs = np.empty((local_dimension, environment_dimension), dtype=np.float64)
    for local in range(local_dimension):
        for environment in range(environment_dimension):
            full = embed_local_bits(local, environment, mask, modes)
            indices[local, environment] = full
            signs[local, environment] = fermion_reorder_sign(full, mask, modes)
    return indices, signs


PAIR_PLANS = {mask: partial_trace_plan(mask) for mask in (0b011, 0b101, 0b110)}


def fermionic_partial_trace(rho: np.ndarray, mask: int) -> np.ndarray:
    indices, signs = PAIR_PLANS[mask]
    out = np.zeros((4, 4), dtype=np.complex128)
    for environment in range(indices.shape[1]):
        index = indices[:, environment]
        sign = signs[:, environment]
        out += rho[index[:, None], index[None, :]] * sign[:, None] * sign[None, :]
    return out


def pair_metrics(rho: np.ndarray) -> dict[str, object]:
    diagonal = np.diag(rho).real
    p_even = float(diagonal[0] + diagonal[3])
    p_odd = float(diagonal[1] + diagonal[2])
    # Basis bits are local modes 0 and 1; this matches dist_scaling.exe.
    g = complex(rho[2, 1])
    f = complex(-rho[3, 0])
    g2 = abs(g) ** 2
    f2 = abs(f) ** 2

    denom_g = math.sqrt(p_even * p_even + 4.0 * g2) + p_even
    denom_f = math.sqrt(p_odd * p_odd + 4.0 * f2) + p_odd
    term_g = 2.0 * g2 / denom_g if denom_g > 0.0 else 0.0
    term_f = 2.0 * f2 / denom_f if denom_f > 0.0 else 0.0
    return {
        "fn": term_g + term_f,
        "F": f,
        "G": g,
        "p_even": p_even,
        "p_odd": p_odd,
    }


def parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def scan_outcomes(path: str | Path) -> tuple[dict[tuple[int, int], Candidate], int]:
    path = Path(path)
    required = {
        "realization_id",
        "anchor_role",
        "pair_i",
        "pair_j",
        "third_k",
        "triple_id",
        "triple_site_a",
        "triple_site_b",
        "triple_site_c",
        "pair_zero_tol",
        "fgmn_positive_tol",
        "pair_fn",
        "connected_ij",
        "ik_fn",
        "jk_fn",
        "cut_i",
        "cut_j",
        "cut_k",
        "min_cut",
        "fgmn_method",
        "fgmn_raw",
        "fgmn",
        "fgmn_positive",
        "fgmn_status",
    }
    positive: dict[tuple[int, int], Candidate] = {}
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        column = {name: header.index(name) for name in required}

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            if len(row) != len(header):
                raise ValueError(
                    f"{path}:{row_number} has {len(row)} fields; expected {len(header)}."
                )
            if row[column["anchor_role"]] != "anchor":
                continue
            if not parse_bool(row[column["connected_ij"]]):
                continue
            if not parse_bool(row[column["fgmn_positive"]]):
                continue
            if row[column["fgmn_status"]] != "ok":
                continue

            realization = int(row[column["realization_id"]])
            triple_id = int(row[column["triple_id"]])
            i = int(row[column["pair_i"]])
            j = int(row[column["pair_j"]])
            k = int(row[column["third_k"]])
            sites = tuple(
                int(row[column[name]])
                for name in ("triple_site_a", "triple_site_b", "triple_site_c")
            )
            pair_values = {
                tuple(sorted((i, j))): finite_float(row[column["pair_fn"]], "pair_fn"),
                tuple(sorted((i, k))): finite_float(row[column["ik_fn"]], "ik_fn"),
                tuple(sorted((j, k))): finite_float(row[column["jk_fn"]], "jk_fn"),
            }
            if len(pair_values) != 3:
                raise ValueError(f"{path}:{row_number} does not describe three distinct sites.")

            key = realization, triple_id
            fgmn = finite_float(row[column["fgmn"]], "fgmn")
            fgmn_raw = finite_float(row[column["fgmn_raw"]], "fgmn_raw")
            candidate = Candidate(
                realization_id=realization,
                triple_id=triple_id,
                sites=sites,
                fgmn=fgmn,
                fgmn_raw=fgmn_raw,
                fgmn_method=row[column["fgmn_method"]],
                fgmn_status=row[column["fgmn_status"]],
                pair_values=pair_values,
                pair_zero_tol_recorded=finite_float(
                    row[column["pair_zero_tol"]], "pair_zero_tol"
                ),
                fgmn_positive_tol_recorded=finite_float(
                    row[column["fgmn_positive_tol"]], "fgmn_positive_tol"
                ),
                cut_values={
                    i: finite_float(row[column["cut_i"]], "cut_i"),
                    j: finite_float(row[column["cut_j"]], "cut_j"),
                    k: finite_float(row[column["cut_k"]], "cut_k"),
                },
                min_cut=finite_float(row[column["min_cut"]], "min_cut"),
                anchor_pairs={tuple(sorted((i, j)))},
            )
            previous = positive.get(key)
            if previous is None:
                positive[key] = candidate
            else:
                if previous.sites != candidate.sites:
                    raise ValueError(f"Conflicting sites for key {key} in {path}.")
                previous.anchor_pairs.update(candidate.anchor_pairs)
                # The same triple can be reused by several anchors.  Retain the
                # row whose recomputed maximum pair fN is smallest; values
                # should agree to rounding because the set of three pairs is
                # unchanged.
                if candidate.max_pair_fn < previous.max_pair_fn:
                    candidate.anchor_pairs.update(previous.anchor_pairs)
                    positive[key] = candidate
    return positive, rows


def select_example_keys(candidates: list[Candidate], examples: int) -> list[Candidate]:
    if not candidates or examples <= 0:
        return []
    rankings = [
        # Strongest three-party certificate.
        sorted(candidates, key=lambda x: (-x.fgmn, x.max_pair_fn)),
        # Cleanest pairwise zeros.
        sorted(candidates, key=lambda x: (x.max_pair_fn, -x.fgmn)),
        # Largest scale separation between fGMN and residual pair fN.
        sorted(
            candidates,
            key=lambda x: (
                -math.log10(max(x.fgmn, 1.0e-300))
                + math.log10(max(x.max_pair_fn, 1.0e-300)),
                -x.fgmn,
            ),
        ),
    ]
    chosen: list[Candidate] = []
    seen: set[tuple[int, int]] = set()
    position = 0
    while len(chosen) < examples:
        added = False
        for ranking in rankings:
            if position >= len(ranking):
                continue
            item = ranking[position]
            if item.key not in seen:
                chosen.append(item)
                seen.add(item.key)
                added = True
                if len(chosen) == examples:
                    break
        if position >= max(len(ranking) for ranking in rankings) and not added:
            break
        position += 1
    return chosen


def read_target_matrices(
    info: Rho3Info,
    targets: list[Candidate],
    chunk_records: int,
) -> dict[tuple[int, int], tuple[tuple[int, int, int], np.ndarray, dict[str, float]]]:
    target_keys = {candidate.key for candidate in targets}
    found: dict[tuple[int, int], tuple[tuple[int, int, int], np.ndarray, dict[str, float]]] = {}
    for block in iter_rho3_chunks(info, chunk_records):
        for record in block:
            key = int(record["realization_id"]), int(record["triple_id"])
            if key not in target_keys or key in found:
                continue
            raw = record["rho"].reshape(8, 8, 2)
            matrix = raw[..., 0] + 1j * raw[..., 1]
            trace_raw = complex(np.trace(matrix))
            if not math.isfinite(trace_raw.real) or trace_raw.real <= 0.0:
                raise ValueError(f"RDM {key} has invalid trace {trace_raw}.")
            hermiticity_raw = float(np.max(np.abs(matrix - matrix.conj().T)))
            matrix = 0.5 * (matrix + matrix.conj().T)
            matrix /= np.trace(matrix).real
            diagnostics = {
                "raw_trace_real": trace_raw.real,
                "raw_trace_imag": trace_raw.imag,
                "raw_hermiticity_max": hermiticity_raw,
                "min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
            }
            sites = tuple(int(value) for value in record["sites"])
            found[key] = sites, matrix, diagnostics
        if len(found) == len(target_keys):
            break
    return found


def recompute_with_solver(
    library_path: Path,
    matrices: dict[tuple[int, int], tuple[tuple[int, int, int], np.ndarray, dict[str, float]]],
) -> dict[tuple[int, int], dict[str, object]]:
    """Re-solve fGMN for selected examples through the repo's shared library."""
    library = ctypes.CDLL(str(library_path.resolve()))
    fgmn = library.compute_fgmn_mosek_8x8_status_cpp
    fgmn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    fgmn.restype = ctypes.c_double
    cuts = library.compute_fermionic_cut_negativities_8x8_cpp
    cuts.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    cuts.restype = ctypes.c_int

    results: dict[tuple[int, int], dict[str, object]] = {}
    for key, (_sites, rho, _diagnostics) in matrices.items():
        flat = rho.reshape(-1)
        interleaved = np.empty(2 * flat.size, dtype=np.float64)
        interleaved[0::2] = flat.real
        interleaved[1::2] = flat.imag
        pointer = interleaved.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        status = ctypes.c_int(-1)
        value = float(fgmn(pointer, ctypes.byref(status)))
        cut_array = (ctypes.c_double * 3)()
        cut_status = int(cuts(pointer, cut_array))
        results[key] = {
            "fgmn_raw": value,
            "fgmn_clamped": max(0.0, value) if math.isfinite(value) else math.nan,
            "status": int(status.value),
            "cuts_status": cut_status,
            "cuts": tuple(float(cut_array[index]) for index in range(3)),
        }
    return results


def basis_label(index: int, modes: int) -> str:
    # Conventional ket order is highest local mode on the left.  For sites
    # (a,b,c), binary index 4 is therefore |n_c n_b n_a> = |100>.
    return format(index, f"0{modes}b")


def complex_text(value: complex) -> str:
    return f"{value.real:+.16e}{value.imag:+.16e}j"


def matrix_text(matrix: np.ndarray) -> str:
    return np.array2string(
        matrix,
        precision=16,
        suppress_small=False,
        max_line_width=260,
        formatter={"complex_kind": complex_text},
    )


def write_outputs(
    prefix: Path,
    all_candidates: list[Candidate],
    examples: list[Candidate],
    matrices: dict[tuple[int, int], tuple[tuple[int, int, int], np.ndarray, dict[str, float]]],
    pair_zero_tol: float,
    fgmn_min: float,
    threshold_counts: list[tuple[float, int]],
    outcome_rows: int,
    info: Rho3Info,
    solver_results: dict[tuple[int, int], dict[str, object]],
) -> tuple[Path, Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_candidates.csv")
    report_path = prefix.with_name(prefix.name + "_examples.txt")
    npz_path = prefix.with_name(prefix.name + "_examples.npz")

    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "realization_id",
                "triple_id",
                "site_a",
                "site_b",
                "site_c",
                "fgmn",
                "fgmn_raw",
                "max_pair_fn",
                "pair_1",
                "pair_1_fn",
                "pair_2",
                "pair_2_fn",
                "pair_3",
                "pair_3_fn",
                "min_cut",
                "anchor_pairs",
            ]
        )
        for item in sorted(all_candidates, key=lambda x: (-x.fgmn, x.max_pair_fn)):
            pairs = sorted(item.pair_values.items())
            writer.writerow(
                [
                    item.realization_id,
                    item.triple_id,
                    *item.sites,
                    format(item.fgmn, ".17g"),
                    format(item.fgmn_raw, ".17g"),
                    format(item.max_pair_fn, ".17g"),
                    f"{pairs[0][0][0]}-{pairs[0][0][1]}",
                    format(pairs[0][1], ".17g"),
                    f"{pairs[1][0][0]}-{pairs[1][0][1]}",
                    format(pairs[1][1], ".17g"),
                    f"{pairs[2][0][0]}-{pairs[2][0][1]}",
                    format(pairs[2][1], ".17g"),
                    format(item.min_cut, ".17g"),
                    ";".join(f"{a}-{b}" for a, b in sorted(item.anchor_pairs)),
                ]
            )

    arrays: dict[str, np.ndarray] = {}
    lines = [
        "PAIRWISE-SILENT, POSITIVE-fGMN THREE-MODE SEARCH",
        "",
        "Definition: fGMN > requested cutoff and every two-mode marginal AB, AC, BC",
        "has nonlogarithmic fermionic negativity no larger than the requested pair cutoff.",
        "This is not zero negativity across 1|23 cuts; positive fGMN excludes that.",
        "",
        f"outcome_rows_scanned={outcome_rows}",
        f"rho3_records={info.record_count}",
        f"rho3_trailing_bytes={info.trailing_bytes}",
        f"selected_pair_zero_tol={pair_zero_tol:.17g}",
        f"selected_fgmn_min={fgmn_min:.17g}",
        f"qualifying_unique_triples={len(all_candidates)}",
        "",
        "Unique positive-fGMN anchor triples satisfying each pair-fN cutoff:",
    ]
    for tolerance, count in threshold_counts:
        lines.append(f"  pair_fn <= {tolerance:.1e}: {count}")

    for number, item in enumerate(examples, start=1):
        entry = matrices.get(item.key)
        lines += ["", "=" * 100, "", f"EXAMPLE {number}"]
        if entry is None:
            lines.append(f"RDM record {item.key} was not found in the binary.")
            continue
        binary_sites, rho, diagnostics = entry
        if binary_sites != item.sites:
            raise ValueError(
                f"CSV sites {item.sites} and binary sites {binary_sites} disagree for {item.key}."
            )
        pair_rows = []
        for mask in (0b011, 0b101, 0b110):
            positions = [position for position in range(3) if (mask >> position) & 1]
            pair_sites = tuple(item.sites[position] for position in positions)
            pair = fermionic_partial_trace(rho, mask)
            metrics = pair_metrics(pair)
            csv_fn = item.pair_values[tuple(sorted(pair_sites))]
            pair_rows.append((pair_sites, pair, metrics, csv_fn))

        recomputed_max = max(float(row[2]["fn"]) for row in pair_rows)
        parity = np.array([index.bit_count() & 1 for index in range(8)])
        leakage_mask = parity[:, None] != parity[None, :]
        parity_leakage = float(np.max(np.abs(rho[leakage_mask])))
        lines += [
            f"realization_id={item.realization_id}",
            f"triple_id={item.triple_id}",
            f"sites={item.sites}",
            f"anchor_pairs={sorted(item.anchor_pairs)}",
            f"fGMN={item.fgmn:.17g}",
            f"fGMN_raw={item.fgmn_raw:.17g}",
            f"fGMN_method={item.fgmn_method}",
            f"fGMN_status={item.fgmn_status}",
            f"recorded_fGMN_positive_tol={item.fgmn_positive_tol_recorded:.17g}",
            f"recorded_pair_zero_tol={item.pair_zero_tol_recorded:.17g}",
            f"CSV_max_pair_fN={item.max_pair_fn:.17g}",
            f"RDM_recomputed_max_pair_fN={recomputed_max:.17g}",
            f"one_vs_two_cut_fN_by_site={dict(sorted(item.cut_values.items()))}",
            f"minimum_one_vs_two_cut_fN={item.min_cut:.17g}",
            f"raw_trace={diagnostics['raw_trace_real']:+.17g}{diagnostics['raw_trace_imag']:+.17g}j",
            f"raw_hermiticity_max={diagnostics['raw_hermiticity_max']:.17g}",
            f"normalized_min_eigenvalue={diagnostics['min_eigenvalue']:.17g}",
            f"normalized_parity_leakage_max={parity_leakage:.17g}",
        ]
        solver = solver_results.get(item.key)
        if solver is None:
            lines.append("independent_fGMN_recompute=not_requested; using stored outcome CSV")
        else:
            lines += [
                f"independent_fGMN_raw={float(solver['fgmn_raw']):.17g}",
                f"independent_fGMN_clamped={float(solver['fgmn_clamped']):.17g}",
                f"independent_fGMN_status={int(solver['status'])}",
                f"independent_cut_fN_status={int(solver['cuts_status'])}",
                f"independent_cut_fN_by_local_position={solver['cuts']}",
            ]
        lines += ["", "Pair marginals (CSV fN, independently recomputed fN, F, G):"]
        for pair_sites, pair, metrics, csv_fn in pair_rows:
            lines.append(
                f"  sites={pair_sites}: CSV={csv_fn:.17g}, "
                f"recomputed={float(metrics['fn']):.17g}, "
                f"F={complex_text(complex(metrics['F']))}, "
                f"G={complex_text(complex(metrics['G']))}"
            )
            lines += [
                f"  rho_{pair_sites} in basis |00>,|10>,|01>,|11>:",
                matrix_text(pair),
            ]

        labels = ", ".join(f"|{basis_label(index, 3)}>" for index in range(8))
        lines += [
            "",
            "Full 8x8 RDM.",
            "For sorted sites (a,b,c), bit 0 is site a; displayed ket order is |n_c n_b n_a>.",
            f"basis={labels}",
            matrix_text(rho),
        ]

        tag = f"example_{number - 1}"
        arrays[f"{tag}_rho3"] = rho
        arrays[f"{tag}_sites"] = np.asarray(item.sites, dtype=np.int64)
        arrays[f"{tag}_metadata"] = np.asarray(
            [
                item.realization_id,
                item.triple_id,
                item.fgmn,
                item.max_pair_fn,
                recomputed_max,
                item.min_cut,
            ],
            dtype=np.float64,
        )
        for pair_number, (_pair_sites, pair, _metrics, _csv_fn) in enumerate(pair_rows):
            arrays[f"{tag}_pair_{pair_number}"] = pair

    report_path.write_text("\n".join(lines) + "\n")
    arrays["metadata_columns"] = np.asarray(
        [
            "realization_id",
            "triple_id",
            "fgmn",
            "csv_max_pair_fn",
            "recomputed_max_pair_fn",
            "min_one_vs_two_cut_fn",
        ]
    )
    arrays["search_settings_json"] = np.asarray(
        json.dumps(
            {
                "pair_zero_tol": pair_zero_tol,
                "fgmn_min": fgmn_min,
                "qualifying_unique_triples": len(all_candidates),
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(npz_path, **arrays)
    return summary_path, report_path, npz_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("rho3_bin", type=Path)
    parser.add_argument("--pair-zero-tol", type=float, default=1.0e-12)
    parser.add_argument("--fgmn-min", type=float, default=1.0e-10)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--chunk-records", type=int, default=32768)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("pairwise_silent_fgmn")
    )
    parser.add_argument(
        "--solver-lib",
        type=Path,
        default=None,
        help=(
            "Optional path to libmipt_fgmn.so. If supplied, independently re-solves "
            "fGMN and the three 1|23 cut negativities for the printed examples."
        ),
    )
    args = parser.parse_args(argv)
    if args.pair_zero_tol < 0.0 or not math.isfinite(args.pair_zero_tol):
        parser.error("--pair-zero-tol must be a non-negative finite number")
    if args.fgmn_min < 0.0 or not math.isfinite(args.fgmn_min):
        parser.error("--fgmn-min must be a non-negative finite number")
    if args.examples < 1:
        parser.error("--examples must be at least 1")
    if args.chunk_records < 1:
        parser.error("--chunk-records must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    info = read_rho3_info(args.rho3_bin)
    if info.trailing_bytes:
        print(
            f"Warning: {args.rho3_bin} has {info.trailing_bytes} trailing bytes; "
            "only complete records will be read.",
            file=sys.stderr,
        )

    print(f"Scanning outcome CSV: {args.outcomes_csv}")
    positive, outcome_rows = scan_outcomes(args.outcomes_csv)
    positive_values = [
        item for item in positive.values() if item.fgmn > args.fgmn_min
    ]
    threshold_grid = sorted(
        {
            0.0,
            1.0e-30,
            1.0e-24,
            1.0e-18,
            1.0e-15,
            1.0e-12,
            args.pair_zero_tol,
        }
    )
    threshold_counts = [
        (tolerance, sum(item.max_pair_fn <= tolerance for item in positive_values))
        for tolerance in threshold_grid
    ]
    qualifying = [
        item for item in positive_values if item.max_pair_fn <= args.pair_zero_tol
    ]

    print(f"Outcome rows scanned: {outcome_rows}")
    print(f"Unique positive-fGMN anchor triples: {len(positive_values)}")
    for tolerance, count in threshold_counts:
        print(f"  all three pair fN <= {tolerance:.1e}: {count}")
    print(
        f"Qualifying at requested cutoffs: {len(qualifying)} "
        f"(pair fN <= {args.pair_zero_tol:.1e}, fGMN > {args.fgmn_min:.1e})"
    )

    examples = select_example_keys(qualifying, args.examples)
    print(
        f"Streaming {info.record_count} RDM records for {len(examples)} selected example(s): "
        f"{args.rho3_bin}"
    )
    matrices = read_target_matrices(info, examples, args.chunk_records)
    missing = [item.key for item in examples if item.key not in matrices]
    if missing:
        print(f"Warning: missing binary records for keys: {missing}", file=sys.stderr)

    solver_results: dict[tuple[int, int], dict[str, object]] = {}
    if args.solver_lib is not None:
        if not args.solver_lib.is_file():
            raise FileNotFoundError(f"Solver library not found: {args.solver_lib}")
        print(f"Independently re-solving {len(matrices)} example fGMN value(s).")
        solver_results = recompute_with_solver(args.solver_lib, matrices)

    paths = write_outputs(
        args.output_prefix,
        qualifying,
        examples,
        matrices,
        args.pair_zero_tol,
        args.fgmn_min,
        threshold_counts,
        outcome_rows,
        info,
        solver_results,
    )
    for path in paths:
        print(f"Wrote {path}")
    if not qualifying:
        print(
            "No qualifying state was found. This is a valid negative result at the chosen "
            "cutoffs; inspect the tighter-threshold counts above before changing them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

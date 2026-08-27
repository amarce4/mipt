"""Reduce ``dist_scaling.exe`` record binaries to small summary CSVs.

    python -m data_analysis.summarize_records <path-or-glob> [...]

or, from the repository root, ``./summarize_records <path-or-glob>``.

Why this exists
---------------
A production record file is far larger than the machine that plots it. A k=2
run at N=24 writes 276 records per trajectory, so 243k trajectories is 67
million records: 2.7 GB on disk, and 4.3 GB more once pandas has widened the
identifiers and given every row its own copy of ``d``. On a box with a few
gigabytes of RAM, loading that is not slow -- it is an out-of-memory kill.

This makes **one streaming pass** in bounded memory and writes a summary of a
few thousand rows: per-geometry sums, sums of squares, NaN and zero counts,
and a logarithmic histogram of ``|value|``. From that,

* the decay curves come back **exactly** -- the mean and standard error are
  reconstructed from the moments, not re-estimated from bins;
* the value distributions come back at the histogram's resolution, which is
  finer than anything they are drawn at.

``data_analysis.dist_scaling`` picks the summary up automatically when it
sits beside the binary, so the expensive pass happens once.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from .records import (
    default_summary_path,
    is_record_file,
    record_file_info,
    summarize,
    write_summary_csv,
)


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        if not matches:
            raise SystemExit(f"No file matched {pattern!r}.")
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.glob("*_records.bin")))
            else:
                paths.append(path)
    if not paths:
        raise SystemExit("Nothing to summarize.")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize_records",
        description=(
            "Stream dist_scaling.exe record binaries and write small summary "
            "CSVs beside them."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="record binaries, globs, or directories holding *_records.bin",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "output CSV (only with a single input; defaults to "
            "<stem>_summary.csv beside each binary)"
        ),
    )
    parser.add_argument(
        "--chunk-records",
        type=int,
        default=1_000_000,
        help=(
            "records held in memory at once (default 1000000, about 40 MB for "
            "a four-observable record). Lower it on a very small machine."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="rewrite a summary that is already newer than its binary",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="no progress output"
    )
    arguments = parser.parse_args(argv)

    paths = _expand(arguments.paths)
    if arguments.output is not None and len(paths) != 1:
        raise SystemExit("--output takes a single input file.")

    for path in paths:
        if not is_record_file(path):
            print(f"skipping {path}: not a dist_scaling record binary", file=sys.stderr)
            continue
        destination = (
            Path(arguments.output)
            if arguments.output is not None
            else default_summary_path(path)
        )
        if (
            not arguments.force
            and destination.exists()
            and destination.stat().st_mtime >= path.stat().st_mtime
        ):
            print(f"{destination} is already up to date (use --force to rewrite).")
            continue

        info = record_file_info(path)
        if not arguments.quiet:
            size = Path(path).stat().st_size
            print(
                f"{path}\n"
                f"  k={info['k']} N={info['L']} "
                f"observables={','.join(info['observables'])} "
                f"records={info['record_count']:,} ({size / 1e9:.2f} GB)"
            )
            if info["truncated"]:
                print(
                    f"  NOTE: {info['truncated']} trailing byte(s) do not form a "
                    "whole record and are ignored; the run was probably killed "
                    "mid-write."
                )
        started = time.monotonic()
        summary = summarize(
            path, chunk_records=arguments.chunk_records, progress=not arguments.quiet
        )
        write_summary_csv(summary, destination)
        if not arguments.quiet:
            elapsed = time.monotonic() - started
            rows = sum(1 for _ in destination.open()) - 1
            print(
                f"  -> {destination} ({rows:,} rows, "
                f"{destination.stat().st_size / 1e3:.0f} kB) in {elapsed:.1f} s "
                f"over {summary.trajectories:,} trajectories"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

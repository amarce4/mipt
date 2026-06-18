from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from parallel_mipt_rho3_writer_safe import write_mipt_rho3_bin_parallel_safe
from mipt import get_mipt_rho_1d
from fermion import get_mipt_rho_1d_ferm
from fermionic_partial_trace import make_tmi_block_plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python MIPT TMI .bin files compatible with the patched C++ tmi.exe."
    )
    parser.add_argument(
        "circuit_type",
        type=int,
        choices=(0, 1),
        help="0 = qubit circuit, 1 = fermionic circuit",
    )
    parser.add_argument("n", type=int, help="Number of qubits/modes. Must be >= 4 and divisible by 4 for TMI.")
    parser.add_argument("realisations", type=int, help="Number of realizations per p value.")
    parser.add_argument("res", type=int, help="Number of p values.")
    parser.add_argument("p_min", type=float, help="Minimum measurement rate.")
    parser.add_argument("p_max", type=float, help="Maximum measurement rate.")
    parser.add_argument("tmi", type=bool, help="Save matrix formatted for TMI (True) or GMN (False)")
    parser.add_argument("--gpu-workers", type=int, default=0, help="Number of GPU worker processes.")
    parser.add_argument("--cpu-workers", type=int, default=4, help="Number of CPU worker processes.")
    parser.add_argument("--out-dir", default="bin", help="Output directory.")
    parser.add_argument("--queue-size", type=int, default=64, help="Multiprocessing queue size.")
    parser.add_argument("--stall-timeout-s", type=float, default=300.0, help="No-result stall timeout in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n = int(args.n)
    d = 2 * n
    realisations = int(args.realisations)
    res = int(args.res)
    ps = np.linspace(float(args.p_min), float(args.p_max), res)

    if n < 4:
        raise ValueError("MIPT circuits must have n >= 4.")
    if (n % 4) != 0 and args.tmi == True:
        raise ValueError("TMI output requires n >= 4 and divisible by 4.")
    if realisations <= 0:
        raise ValueError("realisations must be positive.")
    if res <= 0:
        raise ValueError("res must be positive.")

    is_fermion = bool(args.circuit_type)
    rho_getter = get_mipt_rho_1d_ferm if is_fermion else get_mipt_rho_1d
    plans = make_tmi_block_plans(n, basis_order="little") if is_fermion else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "fermion" if is_fermion else "qubit"
    if args.tmi:
        output_file = out_dir / f"rho_tmi_python_{kind}_n_{n}_real_{realisations}_res_{res}.bin"
    else:
        output_file = out_dir / f"rho_python_{kind}_n_{n}_real_{realisations}_res_{res}.bin"

    print(
        f"Writing {kind} matrices: n={n}, d={d}, realisations={realisations}, "
        f"res={res}, p=[{ps[0]}, {ps[-1]}] -> {output_file}",
        flush=True,
    )

    result = write_mipt_rho3_bin_parallel_safe(
        output_file,
        rho_getter,
        n=n,
        d=d,
        plans=plans,
        p_vals=ps,
        realisations=realisations,
        closed=True,
        tmi=bool(args.tmi),
        gpu_workers=int(args.gpu_workers),
        cpu_workers=int(args.cpu_workers),
        queue_size=int(args.queue_size),
        stall_timeout_s=float(args.stall_timeout_s),
        progress=True,
    )

    print(result)

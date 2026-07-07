from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from mipt import get_mipt_rho_1d, gmn as gmn_score, qubit_partial_trace_keep_density_matrix
from fermion import get_mipt_rho_1d_ferm
from fermionic_partial_trace import (
    fermionic_partial_trace_density_matrix,
    make_ring_3mode_plans,
    make_tmi_block_plans,
    von_neumann_entropy,
)


Metric = Literal["tmi", "gmn"]

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_PLANS = None


def str2bool(value: str | bool) -> bool:
    """argparse-safe bool parser; bool("False") is True, so type=bool is unsafe."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value: 0/1, true/false, yes/no")


def _seed_process(seed: int) -> None:
    """Seed the RNGs used directly or indirectly by the circuit generators."""
    seed32 = int(seed) % (2**32 - 1)
    random.seed(seed32)
    np.random.seed(seed32)

    # Qiskit versions differ in where the global RNG helper lives.  This is a
    # best-effort seed for random_unitary(...) in the qubit path.
    try:  # qiskit >= 1.x
        from qiskit.utils import algorithm_globals  # type: ignore

        algorithm_globals.random_seed = seed32
    except Exception:
        pass


def _as_density_array(rho) -> np.ndarray:
    """Convert Qiskit DensityMatrix or ndarray-like inputs to complex ndarray."""
    data = getattr(rho, "data", rho)
    return np.asarray(data, dtype=np.complex128)


def _entropy(rho) -> float:
    """Numerically stable von Neumann entropy in bits for ndarray/Qiskit matrices."""
    return von_neumann_entropy(_as_density_array(rho), base=2.0, hermitize=True, renormalize=False)


def _qubit_partial_trace_keep(rho, keep: Sequence[int], n_qubits: int) -> np.ndarray:
    """
    Ordinary qubit partial trace in the same local subsystem convention used by
    mipt.py after qc.save_statevector(...).  This keeps secondary reductions
    AB -> A/B and AC -> C convention-locked to get_mipt_rho_1d.
    """
    return qubit_partial_trace_keep_density_matrix(
        _as_density_array(rho),
        keep=keep,
        n_qubits=int(n_qubits),
    )


def tmi_from_block_rhos_qubit(rhos: Sequence[np.ndarray], n: int) -> float:
    """
    Compute I3(A:B:C) from stored qubit TMI blocks in AB, AC, BC, D order.

    For a pure trajectory on ABCD, S_ABC = S_D, so the stored four blocks are
    sufficient:
        I3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_D.
    """
    if len(rhos) != 4:
        raise ValueError(f"TMI requires four block matrices AB, AC, BC, D; got {len(rhos)}.")

    block = int(n) // 4
    rho_ab, rho_ac, rho_bc, rho_d = [_as_density_array(rho) for rho in rhos]

    rho_a = _qubit_partial_trace_keep(rho_ab, range(0, block), 2 * block)
    rho_b = _qubit_partial_trace_keep(rho_ab, range(block, 2 * block), 2 * block)
    rho_c = _qubit_partial_trace_keep(rho_ac, range(block, 2 * block), 2 * block)

    return (
        _entropy(rho_a)
        + _entropy(rho_b)
        + _entropy(rho_c)
        - _entropy(rho_ab)
        - _entropy(rho_ac)
        - _entropy(rho_bc)
        + _entropy(rho_d)
    )


def tmi_from_block_rhos_fermion(rhos: Sequence[np.ndarray], n: int) -> float:
    """
    Compute I3(A:B:C) from fermionically reduced blocks in AB, AC, BC, D order.

    The secondary reductions AB -> A/B and AC -> C are also fermionic partial
    traces, preserving the mode-reordering signs inside the local block basis.
    """
    if len(rhos) != 4:
        raise ValueError(f"TMI requires four block matrices AB, AC, BC, D; got {len(rhos)}.")

    block = int(n) // 4
    rho_ab, rho_ac, rho_bc, rho_d = [_as_density_array(rho) for rho in rhos]

    rho_a = fermionic_partial_trace_density_matrix(
        rho_ab,
        keep=range(0, block),
        n_modes=2 * block,
        basis_order="little",
    )
    rho_b = fermionic_partial_trace_density_matrix(
        rho_ab,
        keep=range(block, 2 * block),
        n_modes=2 * block,
        basis_order="little",
    )
    rho_c = fermionic_partial_trace_density_matrix(
        rho_ac,
        keep=range(block, 2 * block),
        n_modes=2 * block,
        basis_order="little",
    )

    return (
        _entropy(rho_a)
        + _entropy(rho_b)
        + _entropy(rho_c)
        - _entropy(rho_ab)
        - _entropy(rho_ac)
        - _entropy(rho_bc)
        + _entropy(rho_d)
    )


def _mosek_solver_options(mosek_threads: int | None) -> dict[str, Any] | None:
    if mosek_threads is None:
        return None
    return {"mosek_params": {"MSK_IPAR_NUM_THREADS": int(mosek_threads)}}


def _init_worker(config: dict[str, Any]) -> None:
    """Build heavy fermionic trace plans once per worker process."""
    global _WORKER_CONFIG, _WORKER_PLANS
    _WORKER_CONFIG = dict(config)

    n = int(_WORKER_CONFIG["n"])
    is_fermion = bool(_WORKER_CONFIG["is_fermion"])
    tmi_mode = bool(_WORKER_CONFIG["tmi"])

    if is_fermion:
        if tmi_mode:
            _WORKER_PLANS = make_tmi_block_plans(n, basis_order="little")
        else:
            _WORKER_PLANS = make_ring_3mode_plans(n, closed=True, basis_order="little")
    else:
        _WORKER_PLANS = None


def _compute_one(task: tuple[int, int, float, int]) -> list[tuple[float, float]]:
    """
    Compute all CSV rows for one realization.

    TMI returns one row.  GMN returns one row per ring 3-site subsystem, matching
    csim.py's old behavior of storing all 3-site matrices in one record.
    """
    if _WORKER_CONFIG is None:
        raise RuntimeError("worker configuration was not initialized")

    p_index, realization, p, seed = task
    del p_index, realization  # retained in the task for debuggability/order symmetry

    cfg = _WORKER_CONFIG
    n = int(cfg["n"])
    d = int(cfg["d"])
    tmi_mode = bool(cfg["tmi"])
    is_fermion = bool(cfg["is_fermion"])
    use_gpu = bool(cfg["use_gpu"])
    transpile_optimization_level = int(cfg["transpile_optimization_level"])

    _seed_process(seed)

    if is_fermion:
        rhos = get_mipt_rho_1d_ferm(
            n=n,
            d=d,
            p=float(p),
            plans=_WORKER_PLANS,
            subsyst=3,
            all_matrices=not tmi_mode,
            closed=True,
            gpu=use_gpu,
            seed=int(seed),
            transpile_optimization_level=transpile_optimization_level,
            tmi=tmi_mode,
        )
    else:
        rhos = get_mipt_rho_1d(
            n=n,
            d=d,
            p=float(p),
            subsyst=3,
            all_matrices=not tmi_mode,
            closed=True,
            gpu=use_gpu,
            tmi=tmi_mode,
            transpile_optimization_level=transpile_optimization_level,
        )

    if tmi_mode:
        if not isinstance(rhos, list):
            raise ValueError("TMI mode expected a list of four reduced matrices.")
        value = tmi_from_block_rhos_fermion(rhos, n) if is_fermion else tmi_from_block_rhos_qubit(rhos, n)
        return [(float(p), float(value))]

    rho_list = rhos if isinstance(rhos, list) else [rhos]
    solver_options = _mosek_solver_options(cfg.get("mosek_threads"))
    out: list[tuple[float, float]] = []
    for rho in rho_list:
        out.append(
            (
                float(p),
                float(
                    gmn_score(
                        _as_density_array(rho),
                        solver_options=solver_options,
                    )
                ),
            )
        )
    return out


def _iter_tasks(ps: np.ndarray, realisations: int, seed: int | None) -> Iterable[tuple[int, int, float, int]]:
    rng = np.random.default_rng(seed)
    uint32_max = np.iinfo(np.uint32).max
    for p_index, p in enumerate(ps):
        for realization in range(int(realisations)):
            child_seed = int(rng.integers(0, uint32_max))
            yield int(p_index), int(realization), float(p), child_seed


def write_metric_csv(
    output_file: str | os.PathLike[str],
    *,
    circuit_type: int,
    n: int,
    d: int,
    p_vals: Sequence[float],
    realisations: int,
    tmi: bool,
    gpu_workers: int = 0,
    cpu_workers: int = 1,
    chunksize: int = 1,
    seed: int | None = None,
    mosek_threads: int | None = 1,
    transpile_optimization_level: int = 0,
    progress: bool = True,
) -> int:
    """Run the scan and write p,tmi or p,gmn rows. Returns rows written."""
    n = int(n)
    d = int(d)
    realisations = int(realisations)
    p_arr = np.asarray(p_vals, dtype=float)

    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError("p_vals must be a non-empty 1D sequence.")
    if realisations <= 0:
        raise ValueError("realisations must be positive.")
    if n < 4:
        raise ValueError("MIPT circuits must have n >= 4.")
    if tmi and (n % 4) != 0:
        raise ValueError("TMI output requires n >= 4 and divisible by 4.")
    if not tmi and n < 3:
        raise ValueError("GMN output requires n >= 3.")

    gpu_workers = int(gpu_workers)
    cpu_workers = int(cpu_workers)
    if gpu_workers < 0 or cpu_workers < 0:
        raise ValueError("gpu_workers and cpu_workers must be non-negative.")
    if gpu_workers > 0 and cpu_workers > 0:
        raise ValueError(
            "This direct-CSV script supports CPU-only or GPU-only execution per run. "
            "Use either --gpu-workers N --cpu-workers 0 or --gpu-workers 0 --cpu-workers N."
        )

    is_fermion = bool(circuit_type)

    worker_count = gpu_workers if gpu_workers > 0 else cpu_workers
    use_gpu = gpu_workers > 0
    if worker_count <= 0:
        raise ValueError("At least one worker is required.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "tmi" if tmi else "gmn"
    total_tasks = int(p_arr.size) * realisations
    rows_written = 0
    tasks_done = 0
    started = time.perf_counter()
    last_progress = started

    config = {
        "n": n,
        "d": d,
        "is_fermion": bool(circuit_type),
        "tmi": bool(tmi),
        "use_gpu": bool(use_gpu),
        "mosek_threads": mosek_threads,
        "transpile_optimization_level": int(transpile_optimization_level),
    }

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", header])

        if worker_count == 1:
            _init_worker(config)
            for task in _iter_tasks(p_arr, realisations, seed):
                rows = _compute_one(task)
                writer.writerows(rows)
                rows_written += len(rows)
                tasks_done += 1

                now = time.perf_counter()
                if progress and (now - last_progress) >= 5.0:
                    rate = tasks_done / max(now - started, 1e-12)
                    print(
                        f"tasks {tasks_done}/{total_tasks} "
                        f"({100.0 * tasks_done / total_tasks:.1f}%), "
                        f"rows {rows_written}, avg {rate:.2f} tasks/s, ETA {(total_tasks - tasks_done)/rate:.2f}          ",
                        end="\r",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = now
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(config,),
            ) as executor:
                for rows in executor.map(
                    _compute_one,
                    _iter_tasks(p_arr, realisations, seed),
                    chunksize=max(1, int(chunksize)),
                ):
                    writer.writerows(rows)
                    rows_written += len(rows)
                    tasks_done += 1

                    now = time.perf_counter()
                    if progress and (now - last_progress) >= 5.0:
                        rate = tasks_done / max(now - started, 1e-12)
                        print(
                            f"tasks {tasks_done}/{total_tasks} "
                            f"({100.0 * tasks_done / total_tasks:.1f}%), "
                            f"rows {rows_written}, avg {rate:.2f} tasks/s, ETA {(total_tasks - tasks_done)/rate:.2f}          ",
                            end="\r",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_progress = now

    if progress:
        elapsed = time.perf_counter() - started
        print(
            f"\nWrote {rows_written} rows to {output_path} in {elapsed:.2f} s "
            f"({tasks_done / max(elapsed, 1e-12):.2f} tasks/s).",
            file=sys.stderr,
            flush=True,
        )

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python MIPT TMI/GMN CSV files directly, without writing intermediate .bin files."
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
    parser.add_argument("tmi", type=str2bool, help="True/1 = calculate TMI; False/0 = calculate GMN.")
    parser.add_argument("--gpu-workers", type=int, default=0, help="Number of GPU worker processes.")
    parser.add_argument("--cpu-workers", type=int, default=4, help="Number of CPU worker processes.")
    parser.add_argument("--out-dir", default="csv", help="Output directory.")
    parser.add_argument("--output-file", default=None, help="Explicit CSV output path. Overrides --out-dir naming.")
    parser.add_argument("--chunksize", type=int, default=1, help="ProcessPoolExecutor chunksize.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducible task seeding.")
    parser.add_argument(
        "--mosek-threads",
        type=int,
        default=1,
        help="MOSEK threads per GMN solve. Use 1 with process parallelism to avoid oversubscription.",
    )
    parser.add_argument(
        "--transpile-optimization-level",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="Qiskit transpile optimization level passed to the rho getter.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n = int(args.n)
    d = 4 * n
    realisations = int(args.realisations)
    res = int(args.res)
    ps = np.linspace(float(args.p_min), float(args.p_max), res)

    if res <= 0:
        raise ValueError("res must be positive.")

    is_fermion = bool(args.circuit_type)
    kind = "fermion" if is_fermion else "qubit"
    metric: Metric = "tmi" if bool(args.tmi) else "gmn"

    if args.output_file is None:
        out_dir = Path(args.out_dir)
        output_file = out_dir / f"{metric}_python_{kind}_n_{n}_real_{realisations}_res_{res}.csv"
    else:
        output_file = Path(args.output_file)

    print(
        f"Writing {kind} {metric.upper()} CSV: n={n}, d={d}, realisations={realisations}, "
        f"res={res}, p=[{ps[0]}, {ps[-1]}] -> {output_file}",
        flush=True,
    )

    rows = write_metric_csv(
        output_file,
        circuit_type=int(args.circuit_type),
        n=n,
        d=d,
        p_vals=ps,
        realisations=realisations,
        tmi=bool(args.tmi),
        gpu_workers=int(args.gpu_workers),
        cpu_workers=int(args.cpu_workers),
        chunksize=int(args.chunksize),
        seed=args.seed,
        mosek_threads=int(args.mosek_threads),
        transpile_optimization_level=int(args.transpile_optimization_level),
        progress=True,
    )

    print(f"Done: wrote {rows} data rows to {output_file}")

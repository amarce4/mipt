"""
Parallel MIPT rho3 binary writer.

This module runs Python MIPT circuit simulations in multiple worker processes and
serializes all 3-qubit RDMs into one C++-readable MIPTRHO2 .bin file using the
Rho3BinWriter from mipt_rho3_bin_writer.py.

Important constraints
---------------------
1. Only the parent process writes the .bin file. Do not let multiple processes
   write to the same MIPTRHO2 file.
2. CUDA/GPU work should use multiprocessing start method "spawn", not "fork".
3. Usually use at most one GPU worker per physical GPU. More GPU workers often
   make CUDA-Q/state-vector workloads slower or less stable.
4. In notebooks, multiprocessing + CUDA is fragile. Prefer running this from a
   normal .py script guarded by:

       if __name__ == "__main__":
           ...
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mipt_rho3_bin_writer import Rho3BinWriter, ring_subsystem_qubits


@dataclass(frozen=True)
class ParallelRho3Result:
    """Summary returned by write_mipt_rho3_bin_parallel."""

    file: str
    records_written: int
    expected_records: int
    elapsed_s: float
    records_per_s: float


def _worker_loop(
    *,
    worker_id: int,
    use_gpu: bool,
    task_queue,
    result_queue,
    n: int,
    d: int,
    closed: bool,
    cpu_threads_per_worker: Optional[int],
) -> None:
    """Worker process entry point. Kept top-level for spawn pickling."""
    try:
        # Set thread limits before importing mipt/CUDA-Q/numerical libraries in
        # the child. This helps avoid CPU oversubscription when many CPU workers
        # are used.
        if not use_gpu and cpu_threads_per_worker is not None:
            threads = str(int(cpu_threads_per_worker))
            os.environ.setdefault("OMP_NUM_THREADS", threads)
            os.environ.setdefault("MKL_NUM_THREADS", threads)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
            os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)

        from mipt import get_mipt_rho_1d  # Imported inside child process.

        while True:
            task = task_queue.get()
            if task is None:
                return

            p_index, realization, p = task
            rhos = get_mipt_rho_1d(
                int(n),
                int(d),
                float(p),
                subsyst=3,
                all_matrices=True,
                closed=bool(closed),
                gpu=bool(use_gpu),
            )
            result_queue.put(("ok", int(p_index), int(realization), float(p), rhos))

    except BaseException:
        result_queue.put(
            (
                "error",
                int(worker_id),
                bool(use_gpu),
                traceback.format_exc(),
            )
        )


def _start_workers(
    *,
    ctx: BaseContext,
    gpu_workers: int,
    cpu_workers: int,
    task_queue,
    result_queue,
    n: int,
    d: int,
    closed: bool,
    cpu_threads_per_worker: Optional[int],
):
    workers = []
    worker_id = 0

    for _ in range(int(gpu_workers)):
        p = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                worker_id=worker_id,
                use_gpu=True,
                task_queue=task_queue,
                result_queue=result_queue,
                n=n,
                d=d,
                closed=closed,
                cpu_threads_per_worker=cpu_threads_per_worker,
            ),
        )
        p.start()
        workers.append(p)
        worker_id += 1

    for _ in range(int(cpu_workers)):
        p = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                worker_id=worker_id,
                use_gpu=False,
                task_queue=task_queue,
                result_queue=result_queue,
                n=n,
                d=d,
                closed=closed,
                cpu_threads_per_worker=cpu_threads_per_worker,
            ),
        )
        p.start()
        workers.append(p)
        worker_id += 1

    return workers


def write_mipt_rho3_bin_parallel(
    file: str | os.PathLike[str],
    *,
    n: int,
    d: int,
    p_vals: Sequence[float],
    realisations: int,
    closed: bool = True,
    gpu_workers: int = 1,
    cpu_workers: int = 0,
    cpu_threads_per_worker: Optional[int] = 1,
    subsystem_qubits: Optional[Sequence[Sequence[int]]] = None,
    spatial_dimension: int = 1,
    hermitize: bool = False,
    normalize_trace: bool = False,
    check_finite: bool = True,
    queue_size: int = 256,
    start_method: str = "spawn",
    progress: bool = True,
    progress_interval_s: float = 2.0,
) -> ParallelRho3Result:
    """
    Run MIPT simulations in parallel and write one MIPTRHO2 .bin file.

    Parameters
    ----------
    file
        Output path. Existing files are overwritten by Rho3BinWriter.

    n, d, p_vals, realisations, closed
        Same meaning as in the serial simulation loop.

    gpu_workers
        Number of worker processes calling get_mipt_rho_1d(..., gpu=True).
        Use 0 to disable GPU work. For one physical GPU, 1 is usually best.

    cpu_workers
        Number of worker processes calling get_mipt_rho_1d(..., gpu=False).
        Start with 1-4. Too many can oversubscribe CPU threads.

    cpu_threads_per_worker
        Sets OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, and
        NUMEXPR_NUM_THREADS inside CPU workers before importing mipt. Use 1 when
        launching several CPU worker processes. Use None to leave environment
        unchanged.

    queue_size
        Bounded multiprocessing queue size. Larger values use more RAM but can
        reduce idle time.

    start_method
        Must normally be "spawn" for CUDA safety. Do not use "fork" with CUDA.

    Returns
    -------
    ParallelRho3Result
        Summary of record count and throughput.

    Notes
    -----
    Output record order is completion order, not necessarily sorted by p_index
    then realization. The binary format and the C++ GMN program do not require
    sorted records because each record carries p_index, realization, and p.
    """
    p_arr = np.asarray(p_vals, dtype=float)
    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError("p_vals must be a non-empty 1D sequence.")
    if int(realisations) <= 0:
        raise ValueError("realisations must be positive.")
    if int(gpu_workers) < 0 or int(cpu_workers) < 0:
        raise ValueError("gpu_workers and cpu_workers must be non-negative.")
    worker_count = int(gpu_workers) + int(cpu_workers)
    if worker_count <= 0:
        raise ValueError("At least one GPU or CPU worker is required.")

    if subsystem_qubits is None:
        subsystem_qubits = ring_subsystem_qubits(int(n), closed=closed)

    total_records = int(p_arr.size) * int(realisations)
    output_path = Path(file)

    ctx = get_context(start_method)
    task_queue = ctx.Queue(maxsize=int(queue_size))
    result_queue = ctx.Queue(maxsize=int(queue_size))

    def producer() -> None:
        try:
            for p_index, p in enumerate(p_arr):
                for realization in range(int(realisations)):
                    task_queue.put((int(p_index), int(realization), float(p)))
        finally:
            for _ in range(worker_count):
                task_queue.put(None)

    workers = _start_workers(
        ctx=ctx,
        gpu_workers=int(gpu_workers),
        cpu_workers=int(cpu_workers),
        task_queue=task_queue,
        result_queue=result_queue,
        n=int(n),
        d=int(d),
        closed=bool(closed),
        cpu_threads_per_worker=cpu_threads_per_worker,
    )

    producer_thread = threading.Thread(target=producer, name="rho3-task-producer", daemon=True)
    producer_thread.start()

    start = time.perf_counter()
    written = 0
    last_progress = start
    per_p_counts = np.zeros(p_arr.size, dtype=np.int64)

    try:
        with Rho3BinWriter(
            output_path,
            total_qubits=int(n),
            subsystem_qubits=subsystem_qubits,
            spatial_dimension=spatial_dimension,
            periods=int(d),
            realizations=int(realisations),
            p_vals=p_arr,
            grid_y=1,
            hermitize=hermitize,
            normalize_trace=normalize_trace,
            check_finite=check_finite,
        ) as writer:
            while written < total_records:
                try:
                    msg = result_queue.get(timeout=1.0)
                except queue.Empty:
                    # Detect dead workers to avoid hanging forever.
                    dead = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
                    if dead:
                        codes = ", ".join(str(p.exitcode) for p in dead)
                        raise RuntimeError(f"One or more workers exited unexpectedly: {codes}")
                    continue

                tag = msg[0]
                if tag == "error":
                    _, worker_id, use_gpu, tb = msg
                    raise RuntimeError(
                        f"Worker {worker_id} failed "
                        f"({'GPU' if use_gpu else 'CPU'}):\n{tb}"
                    )

                _, p_index, realization, p, rhos = msg
                writer.write_record(
                    p_index=p_index,
                    realization=realization,
                    p=p,
                    rhos=rhos,
                )
                written += 1
                per_p_counts[int(p_index)] += 1

                now = time.perf_counter()
                if progress and (now - last_progress) >= float(progress_interval_s):
                    rate = written / max(now - start, 1e-12)
                    # Show the least-complete p bucket as a compact progress indicator.
                    min_p_done = int(per_p_counts.min())
                    max_p_done = int(per_p_counts.max())
                    print(
                        f"records {written}/{total_records} "
                        f"({100.0 * written / total_records:.1f}%), "
                        f"{rate:.2f} rec/s, "
                        f"per-p range {min_p_done}-{max_p_done}/{realisations}              ",
                        file=sys.stderr,
                        end='\r',
                        flush=True
                    )
                    last_progress = now

    except BaseException:
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=5.0)
        raise
    finally:
        producer_thread.join(timeout=2.0)
        for p in workers:
            p.join(timeout=10.0)

    elapsed = time.perf_counter() - start
    return ParallelRho3Result(
        file=str(output_path),
        records_written=int(written),
        expected_records=int(total_records),
        elapsed_s=float(elapsed),
        records_per_s=float(written / max(elapsed, 1e-12)),
    )

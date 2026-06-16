"""
Safer parallel writer for MIPT rho3 MIPTRHO2 .bin files.

Compared with the earlier helper, this version is designed for long runs:

- The parent process is still the only process that writes the .bin file.
- It reports both average and recent throughput.
- It can stop on a no-result stall instead of hanging forever.
- On KeyboardInterrupt or stall timeout, it can finalize the partial file header
  so the C++ GMN program can still read the records already written.

Recommended first use for CUDA-Q-style workloads:

    result = write_mipt_rho3_bin_parallel_safe(
        "rho3_python_archA.bin",
        n=8,
        d=16,
        p_vals=np.linspace(0.0, 1.0, 21),
        realisations=10000,
        gpu_workers=1,
        cpu_workers=0,
        stall_timeout_s=300,
    )
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mipt_rho3_bin_writer import Rho3BinWriter, ring_subsystem_qubits


@dataclass(frozen=True)
class ParallelRho3SafeResult:
    file: str
    records_written: int
    expected_records: int
    elapsed_s: float
    average_records_per_s: float
    completed: bool
    stopped_reason: str


def _worker_loop(
    *,
    rho_getter,
    worker_id: int,
    use_gpu: bool,
    task_queue,
    result_queue,
    n: int,
    d: int,
    closed: bool,
    cpu_threads_per_worker: Optional[int],
) -> None:
    try:
        if not use_gpu and cpu_threads_per_worker is not None:
            threads = str(int(cpu_threads_per_worker))
            os.environ.setdefault("OMP_NUM_THREADS", threads)
            os.environ.setdefault("MKL_NUM_THREADS", threads)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
            os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)

        # from mipt import get_mipt_rho_1d

        while True:
            task = task_queue.get()
            if task is None:
                return

            p_index, realization, p = task
            rhos = rho_getter(
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
        result_queue.put(("error", int(worker_id), bool(use_gpu), traceback.format_exc()))


def _start_workers(
    *,
    rho_getter,
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
        proc = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                rho_getter=rho_getter,
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
        proc.start()
        workers.append(proc)
        worker_id += 1

    for _ in range(int(cpu_workers)):
        proc = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                rho_getter=rho_getter,
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
        proc.start()
        workers.append(proc)
        worker_id += 1

    return workers


def write_mipt_rho3_bin_parallel_safe(
    file: str | os.PathLike[str],
    rho_getter,
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
    queue_size: int = 128,
    start_method: str = "spawn",
    progress: bool = True,
    progress_interval_s: float = 5.0,
    recent_window_s: float = 60.0,
    stall_timeout_s: Optional[float] = 300.0,
    finalize_partial_on_stop: bool = True,
) -> ParallelRho3SafeResult:
    """
    Run MIPT simulations in parallel and write one MIPTRHO2 .bin file.

    If the run is interrupted or no result arrives for stall_timeout_s seconds,
    the already-written records are finalized into the file header when
    finalize_partial_on_stop=True. This makes the partial file readable by the
    C++ GMN program.
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
        raise ValueError("At least one worker is required.")

    if int(gpu_workers) > 0 and int(cpu_workers) > 0:
        print(
            "Warning: mixing GPU and CPU workers can be slower or less stable for CUDA-Q-style workloads. "
            "For diagnosis, try gpu_workers=1, cpu_workers=0.",
            file=sys.stderr,
            flush=True,
        )

    if subsystem_qubits is None:
        subsystem_qubits = ring_subsystem_qubits(int(n), closed=closed)

    total_records = int(p_arr.size) * int(realisations)
    output_path = Path(file)

    ctx = get_context(start_method)
    task_queue = ctx.Queue(maxsize=int(queue_size))
    result_queue = ctx.Queue(maxsize=int(queue_size))

    producer_exception: list[BaseException] = []

    def producer() -> None:
        try:
            for p_index, p in enumerate(p_arr):
                for realization in range(int(realisations)):
                    task_queue.put((int(p_index), int(realization), float(p)))
        except BaseException as exc:  # pragma: no cover - defensive
            producer_exception.append(exc)
        finally:
            for _ in range(worker_count):
                try:
                    task_queue.put(None)
                except BaseException:
                    pass

    workers = _start_workers(
        rho_getter=rho_getter,
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
    last_progress = start
    last_result_time = start
    written = 0
    stopped_reason = "completed"
    completed = False
    per_p_counts = np.zeros(p_arr.size, dtype=np.int64)
    recent_events: deque[tuple[float, int]] = deque()

    writer = Rho3BinWriter(
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
    )

    try:
        while written < total_records:
            if producer_exception:
                raise RuntimeError(f"Producer failed: {producer_exception[0]!r}")

            try:
                msg = result_queue.get(timeout=1.0)
            except queue.Empty:
                dead_bad = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
                if dead_bad:
                    codes = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in dead_bad)
                    raise RuntimeError(f"One or more workers exited unexpectedly: {codes}")

                now = time.perf_counter()
                if stall_timeout_s is not None and (now - last_result_time) >= float(stall_timeout_s):
                    alive = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in workers)
                    stopped_reason = f"stalled: no result for {stall_timeout_s:g} s; workers: {alive}"
                    raise TimeoutError(stopped_reason)
                continue

            tag = msg[0]
            if tag == "error":
                _, worker_id, use_gpu, tb = msg
                raise RuntimeError(f"Worker {worker_id} failed ({'GPU' if use_gpu else 'CPU'}):\n{tb}")

            _, p_index, realization, p, rhos = msg
            writer.write_record(p_index=p_index, realization=realization, p=p, rhos=rhos)
            written += 1
            per_p_counts[int(p_index)] += 1
            now = time.perf_counter()
            last_result_time = now
            recent_events.append((now, 1))
            while recent_events and (now - recent_events[0][0]) > float(recent_window_s):
                recent_events.popleft()

            if progress and (now - last_progress) >= float(progress_interval_s):
                elapsed = max(now - start, 1e-12)
                avg_rate = written / elapsed
                if recent_events:
                    recent_elapsed = max(now - recent_events[0][0], 1e-12)
                    recent_rate = sum(count for _, count in recent_events) / recent_elapsed
                else:
                    recent_rate = 0.0
                min_p_done = int(per_p_counts.min())
                max_p_done = int(per_p_counts.max())
                print(
                    f"records {written}/{total_records} "
                    f"({100.0 * written / total_records:.1f}%), "
                    f"avg {avg_rate:.2f} rec/s, recent {recent_rate:.2f} rec/s, "
                    f"per-p range {min_p_done}-{max_p_done}/{realisations}              ",
                    file=sys.stderr,
                    flush=True,
                    end='\r'
                )
                last_progress = now

        completed = True
        stopped_reason = "completed"
        writer.close(finalize=True)

    except KeyboardInterrupt:
        stopped_reason = "interrupted by KeyboardInterrupt"
        if finalize_partial_on_stop:
            writer.close(finalize=True)
        else:
            writer.close(finalize=False)
        raise
    except BaseException:
        if finalize_partial_on_stop:
            writer.close(finalize=True)
        else:
            writer.close(finalize=False)
        raise
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=5.0)
        producer_thread.join(timeout=2.0)
        if not writer._closed:  # type: ignore[attr-defined]
            writer.close(finalize=finalize_partial_on_stop)

    elapsed = time.perf_counter() - start
    return ParallelRho3SafeResult(
        file=str(output_path),
        records_written=int(written),
        expected_records=int(total_records),
        elapsed_s=float(elapsed),
        average_records_per_s=float(written / max(elapsed, 1e-12)),
        completed=bool(completed),
        stopped_reason=stopped_reason,
    )

"""
Safer parallel writer for MIPT reduced-density-matrix .bin files.

Supported output formats:

- tmi=False: legacy MIPTRHO2 rho3 files for GMN/rho3 workflows.
- tmi=True:  MIPTTMI1 four-block files for the patched C++ tmi.exe.

The parent process is the only process that writes the .bin file. Worker
processes only generate one trajectory's matrices and return them through a
multiprocessing queue.
"""

from __future__ import annotations

import inspect
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from mipt_rho3_bin_writer import Rho3BinWriter, TmiBinWriter, ring_subsystem_qubits, tmi_subsystem_terms


@dataclass(frozen=True)
class ParallelRho3SafeResult:
    file: str
    records_written: int
    expected_records: int
    elapsed_s: float
    average_records_per_s: float
    completed: bool
    stopped_reason: str


def _filter_kwargs_for_callable(func, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unsupported keyword arguments unless func accepts **kwargs."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)

    params = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return dict(kwargs)

    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _call_rho_getter(
    rho_getter,
    *,
    n: int,
    d: int,
    p: float,
    use_gpu: bool,
    closed: bool,
    tmi: bool,
    plans,
    rho_getter_args: Sequence[Any],
    rho_getter_kwargs: Mapping[str, Any],
):
    """
    Call either get_mipt_rho_1d or get_mipt_rho_1d_ferm without hardcoding the
    full downstream signature.

    Defaults are supplied for the common MIPT rho-getter options, then filtered
    against the callable signature. Explicit rho_getter_kwargs override these
    defaults.
    """
    kwargs: dict[str, Any] = {
        "plans": plans,
        "subsyst": 3,
        "all_matrices": True,
        "closed": bool(closed),
        "gpu": bool(use_gpu),
        "tmi": bool(tmi),
    }
    kwargs.update(dict(rho_getter_kwargs))
    kwargs = _filter_kwargs_for_callable(rho_getter, kwargs)

    return rho_getter(int(n), int(d), float(p), *tuple(rho_getter_args), **kwargs)


def _worker_loop(
    *,
    rho_getter,
    worker_id: int,
    use_gpu: bool,
    task_queue,
    result_queue,
    n: int,
    d: int,
    plans,
    closed: bool,
    tmi: bool,
    rho_getter_args: Sequence[Any],
    rho_getter_kwargs: Mapping[str, Any],
    cpu_threads_per_worker: Optional[int],
) -> None:
    try:
        if not use_gpu and cpu_threads_per_worker is not None:
            threads = str(int(cpu_threads_per_worker))
            os.environ.setdefault("OMP_NUM_THREADS", threads)
            os.environ.setdefault("MKL_NUM_THREADS", threads)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
            os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)

        while True:
            task = task_queue.get()
            if task is None:
                return

            p_index, realization, p = task
            rhos = _call_rho_getter(
                rho_getter,
                n=int(n),
                d=int(d),
                p=float(p),
                use_gpu=bool(use_gpu),
                closed=bool(closed),
                tmi=bool(tmi),
                plans=plans,
                rho_getter_args=rho_getter_args,
                rho_getter_kwargs=rho_getter_kwargs,
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
    plans,
    closed: bool,
    tmi: bool,
    rho_getter_args: Sequence[Any],
    rho_getter_kwargs: Mapping[str, Any],
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
                plans=plans,
                closed=closed,
                tmi=tmi,
                rho_getter_args=tuple(rho_getter_args),
                rho_getter_kwargs=dict(rho_getter_kwargs),
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
                plans=plans,
                closed=closed,
                tmi=tmi,
                rho_getter_args=tuple(rho_getter_args),
                rho_getter_kwargs=dict(rho_getter_kwargs),
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
    plans=None,
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
    tmi: bool = False,
    rho_getter_args: Optional[Sequence[Any]] = None,
    rho_getter_kwargs: Optional[Mapping[str, Any]] = None,
) -> ParallelRho3SafeResult:
    """
    Run MIPT simulations in parallel and write one .bin file.

    Parameters
    ----------
    rho_getter:
        Callable such as get_mipt_rho_1d or get_mipt_rho_1d_ferm. It is called
        as ``rho_getter(n, d, p, *rho_getter_args, **filtered_kwargs)``.
        Common kwargs (plans, subsyst, all_matrices, closed, gpu, tmi) are
        supplied automatically when supported by the callable.

    tmi:
        False writes the legacy MIPTRHO2/rho3 format. True writes the MIPTTMI1
        four-block format consumed by the patched C++ tmi.exe.
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
            "Warning: mixing GPU and CPU workers can be slower or less stable for simulator workloads. "
            "For diagnosis, try gpu_workers=1, cpu_workers=0.",
            file=sys.stderr,
            flush=True,
        )

    rho_getter_args = tuple(rho_getter_args or ())
    rho_getter_kwargs = dict(rho_getter_kwargs or {})

    if not tmi and subsystem_qubits is None:
        subsystem_qubits = ring_subsystem_qubits(int(n), closed=closed)

    total_records = int(p_arr.size) * int(realisations)
    output_path = Path(file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        plans=plans,
        closed=bool(closed),
        tmi=bool(tmi),
        rho_getter_args=rho_getter_args,
        rho_getter_kwargs=rho_getter_kwargs,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )

    producer_thread = threading.Thread(target=producer, name="rho-task-producer", daemon=True)
    producer_thread.start()

    start = time.perf_counter()
    last_progress = start
    last_result_time = start
    written = 0
    stopped_reason = "completed"
    completed = False
    per_p_counts = np.zeros(p_arr.size, dtype=np.int64)
    recent_events: deque[tuple[float, int]] = deque()

    if tmi:
        writer = TmiBinWriter(
            output_path,
            total_qubits=int(n),
            terms=tmi_subsystem_terms(int(n)),
            spatial_dimension=spatial_dimension,
            periods=int(d),
            realizations=int(realisations),
            p_vals=p_arr,
            grid_y=1,
            hermitize=hermitize,
            normalize_trace=normalize_trace,
            check_finite=check_finite,
        )
    else:
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
                    end="\r",
                )
                last_progress = now

        completed = True
        stopped_reason = "completed"
        writer.close(finalize=True)

    except KeyboardInterrupt:
        stopped_reason = "interrupted by KeyboardInterrupt"
        writer.close(finalize=finalize_partial_on_stop)
        raise
    except BaseException:
        writer.close(finalize=finalize_partial_on_stop)
        raise
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=5.0)
        producer_thread.join(timeout=2.0)
        if not getattr(writer, "_closed", True):
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

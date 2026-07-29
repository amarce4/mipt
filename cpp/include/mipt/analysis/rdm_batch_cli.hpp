#pragma once

// Shared driver for the executables that stream an MIPTRHO2 file and solve one
// SDP per retained subsystem: gmn.exe and fgmn.exe.
//
// Both have the same shape — read a batch of trajectory records, expand each
// into per-subsystem jobs, solve the batch in parallel, append the rows, and
// report progress — and differ only in the job payload, the solve, and the
// columns. Those three live in the app; everything else lives here.
//
// A note on file size: these outputs are row-level, one line per subsystem per
// trajectory, and routinely reach millions of rows. Run metadata is therefore
// written to a small sidecar rather than repeated on every row.

#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/util/pause.hpp"
#include "mipt/util/text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mipt::analysis
{

// One SDP evaluation.
struct SolveResult
{
    double value = std::numeric_limits<double>::quiet_NaN();
    double bipneg = std::numeric_limits<double>::quiet_NaN();
    bool prefiltered = false;
};

struct BatchOptions
{
    std::uint64_t limit_per_p = 0;
    int batch_records = 256;
};

struct BatchTotals
{
    std::uint64_t records_scanned = 0;
    std::uint64_t records_evaluated = 0;
    std::uint64_t records_skipped_by_limit = 0;
    std::uint64_t processed = 0;
    std::uint64_t total_solves = 0;
    std::uint64_t zero_prefilter_skipped = 0;
    std::uint64_t nonfinite_values = 0;
};

// "<output>.csv" -> "<output>_meta.csv"
inline std::string metadata_sidecar_path(const std::string &output_path)
{
    const auto dot = output_path.find_last_of('.');
    const auto slash = output_path.find_last_of("/\\");
    const bool has_extension = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return has_extension ? output_path.substr(0, dot) + "_meta.csv" : output_path + "_meta.csv";
}

// Self-describing companion to the row-level CSV. The MIPTRHO2 header does not
// record the circuit type, so that field is intentionally absent here rather
// than guessed.
inline void write_metadata_sidecar(const std::string &output_path, const io::DensityFileMetadata &metadata,
                                   const std::string &input_path, const std::string &solver)
{
    const std::string path = metadata_sidecar_path(output_path);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
        std::cerr << "Warning: could not write the metadata sidecar " << path << ".\n";
        return;
    }
    out << std::setprecision(17);
    out << "N,spatial_dimension,grid_x,grid_y,periods,realizations,resolution,"
           "p_min,p_max,kept_qubits,subsystem_count,record_count,solver,source\n";
    out << metadata.total_qubits << ',' << metadata.spatial_dimension << ',' << metadata.grid_x << ','
        << metadata.grid_y << ',' << metadata.periods << ',' << metadata.realizations << ',' << metadata.resolution
        << ',' << metadata.p_min << ',' << metadata.p_max << ',' << metadata.kept_qubits << ','
        << metadata.subsystem_count << ',' << metadata.record_count << ',' << util::csv_quote(solver) << ','
        << util::csv_quote(input_path) << '\n';
    if (!out)
    {
        std::cerr << "Warning: failed while writing the metadata sidecar " << path << ".\n";
    }
}

inline void require_three_qubit_input(const io::DensityFileMetadata &metadata)
{
    if (metadata.kept_qubits != 3 || metadata.matrix_dimension != 8)
    {
        throw std::runtime_error("Input file must contain three-qubit 8x8 reduced density matrices.");
    }
}

// How many records will actually be evaluated once limit_per_p is applied.
inline std::uint64_t solves_to_perform(const io::DensityFileMetadata &metadata, std::uint64_t limit_per_p)
{
    std::uint64_t max_records = metadata.record_count;
    if (limit_per_p != 0)
    {
        const auto resolution = static_cast<std::uint64_t>(metadata.resolution);
        const std::uint64_t limited =
            (resolution != 0 && limit_per_p > std::numeric_limits<std::uint64_t>::max() / resolution)
                ? std::numeric_limits<std::uint64_t>::max()
                : limit_per_p * resolution;
        max_records = std::min<std::uint64_t>(metadata.record_count, limited);
    }
    return max_records * static_cast<std::uint64_t>(metadata.subsystem_count);
}

inline void print_worker_configuration(const char *measure)
{
#ifdef _OPENMP
    std::cout << "OpenMP " << measure << " workers: " << omp_get_max_threads() << "; GMN_MOSEK_NUM_THREADS="
              << (std::getenv("GMN_MOSEK_NUM_THREADS") ? std::getenv("GMN_MOSEK_NUM_THREADS") : "MOSEK default")
              << ".\n";
#else
    std::cout << "OpenMP is not enabled in this build; " << measure << " solves are serial.\n";
#endif
}

// Parallelism is at the independent-SDP level, so each MOSEK solve should stay
// single-threaded unless the user asked otherwise.
inline void limit_nested_solver_threads()
{
#ifdef _OPENMP
    if (omp_get_max_threads() > 1)
    {
        env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
    }
#endif
}

namespace detail
{

inline void report_progress(const BatchTotals &totals, const std::chrono::steady_clock::time_point &start,
                            const std::chrono::steady_clock::time_point &interval_start,
                            std::uint64_t interval_processed, bool show_prefilter)
{
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = now - start;
    const std::chrono::duration<double> interval_elapsed = now - interval_start;
    const double average_rate = elapsed.count() > 0.0 ? totals.processed / elapsed.count() : 0.0;
    const double recent_rate =
        interval_elapsed.count() > 0.0 ? interval_processed / interval_elapsed.count() : 0.0;
    const double eta_seconds = average_rate > 0.0 ? (totals.total_solves - totals.processed) / average_rate : 0.0;

    std::cout << "\rProcessed " << totals.processed << "/" << totals.total_solves << " matrices from "
              << totals.records_evaluated << " records";
    if (totals.records_skipped_by_limit != 0)
    {
        std::cout << " (skipped " << totals.records_skipped_by_limit << " by limit)";
    }
    if (show_prefilter && totals.zero_prefilter_skipped != 0)
    {
        std::cout << "; zero-prefilter " << totals.zero_prefilter_skipped;
    }
    std::cout << "; avg " << std::fixed << std::setprecision(2) << average_rate << "/s, recent " << recent_rate
              << "/s, ETA " << eta_seconds << " s        " << std::flush;
}

} // namespace detail

// Stream the whole input in batches.
//
//   append_jobs(metadata, record, jobs)  expand one trajectory into jobs
//   solve_job(job)                       one SDP, called from an OpenMP loop
//   write_row(out, job, result)          one CSV line
template <typename Job, typename AppendJobs, typename SolveJob, typename WriteRow>
BatchTotals run_rdm_batches(io::DensityMatrixReader &reader, std::ostream &out, const BatchOptions &options,
                            AppendJobs append_jobs, SolveJob solve_job, WriteRow write_row,
                            bool show_prefilter_progress = false)
{
    const auto &metadata = reader.metadata();
    BatchTotals totals;
    totals.total_solves = solves_to_perform(metadata, options.limit_per_p);

    std::vector<Job> jobs;
    jobs.reserve(static_cast<std::size_t>(options.batch_records) * metadata.subsystem_count);
    std::vector<SolveResult> results;

    io::DensityRecord record;
    std::vector<std::uint64_t> accepted_records_per_p(metadata.resolution, 0);
    bool reader_exhausted = false;
    util::PauseSentinel pause_sentinel;
    const auto start = std::chrono::steady_clock::now();
    auto interval_start = start;
    std::uint64_t interval_processed = 0;

    while (totals.processed < totals.total_solves && !reader_exhausted)
    {
        // Safe checkpoint: the previous batch has been written and flushed.
        pause_sentinel.wait([&]() { out.flush(); });

        jobs.clear();

        int records_in_batch = 0;
        while (records_in_batch < options.batch_records)
        {
            if (!reader.read_record(record))
            {
                reader_exhausted = true;
                break;
            }

            ++totals.records_scanned;
            ++records_in_batch;

            if (record.p_index >= accepted_records_per_p.size())
            {
                throw std::runtime_error("Input record has p_index outside metadata.resolution.");
            }
            if (options.limit_per_p != 0 && accepted_records_per_p[record.p_index] >= options.limit_per_p)
            {
                ++totals.records_skipped_by_limit;
                continue;
            }

            ++accepted_records_per_p[record.p_index];
            ++totals.records_evaluated;
            append_jobs(metadata, record, jobs);
        }

        if (jobs.empty())
        {
            if (!reader_exhausted)
            {
                std::cout << "\rScanned " << totals.records_scanned << " records; skipped "
                          << totals.records_skipped_by_limit
                          << " by limit_per_p; waiting for the next accepted p-record...        " << std::flush;
                continue;
            }
            break;
        }

        results.assign(jobs.size(), SolveResult{});

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(jobs.size()); ++i)
        {
            const auto index = static_cast<std::size_t>(i);
            results[index] = solve_job(jobs[index]);
        }

        for (std::size_t i = 0; i < jobs.size(); ++i)
        {
            if (results[i].prefiltered)
            {
                ++totals.zero_prefilter_skipped;
            }
            if (!std::isfinite(results[i].value))
            {
                ++totals.nonfinite_values;
            }
            write_row(out, jobs[i], results[i]);
        }
        out.flush();

        totals.processed += jobs.size();
        interval_processed += jobs.size();
        detail::report_progress(totals, start, interval_start, interval_processed, show_prefilter_progress);
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - interval_start).count() >= 5.0)
        {
            interval_start = std::chrono::steady_clock::now();
            interval_processed = 0;
        }
    }
    return totals;
}

inline void report_batch_summary(const std::string &output_path, const BatchTotals &totals, double max_imag_norm,
                                 const char *imag_norm_label)
{
    std::cout << "\nWrote " << output_path << ".\n"
              << "Records scanned: " << totals.records_scanned
              << "; records evaluated: " << totals.records_evaluated
              << "; records skipped by limit_per_p: " << totals.records_skipped_by_limit << "\n"
              << "Matrices skipped by bipartite-negativity zero prefilter: " << totals.zero_prefilter_skipped << "\n"
              << imag_norm_label << ": " << std::scientific << std::setprecision(6) << max_imag_norm << "\n";
}

} // namespace mipt::analysis

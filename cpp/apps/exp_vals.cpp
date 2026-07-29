#include "mipt/analysis/exp_vals.hpp"
#include "mipt/density.hpp"
#include "mipt/util/pause.hpp"
#include "mipt/util/text.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    constexpr std::size_t MODE_COUNT = 3;
    constexpr std::size_t MATRIX_DIMENSION = 8;
    constexpr std::size_t MATRIX_VALUE_COUNT =
        2 * MATRIX_DIMENSION * MATRIX_DIMENSION;
    constexpr std::size_t DEFAULT_BATCH_RECORDS = 256;
    constexpr double DEFAULT_PROGRESS_SECONDS = 2.0;

    enum Metric : std::size_t
    {
        PURITY,
        HOP_R1_SQ,
        HOP_R2_SQ,
        PAIR_R1_SQ,
        PAIR_R2_SQ,
        NUMBER_MEAN,
        NUMBER_VAR,
        PARITY_EXPECTATION,
        PARITY_VAR,
        DENSITY_R1,
        DENSITY_R2,
        DENSITY_R1_SQ,
        DENSITY_R2_SQ,
        WICK4,
        WICK6,
        METRIC_COUNT
    };

    constexpr std::array<const char *, METRIC_COUNT> METRIC_NAMES{{
        "purity",
        "hop_r1_sq",
        "hop_r2_sq",
        "pair_r1_sq",
        "pair_r2_sq",
        "number_expectation",
        "number_variance",
        "parity_expectation",
        "parity_variance",
        "density_r1",
        "density_r2",
        "density_r1_sq",
        "density_r2_sq",
        "wick4",
        "wick6",
    }};

    using MetricArray = std::array<double, METRIC_COUNT>;

    struct TrajectorySummary
    {
        std::uint32_t p_index = 0;
        double p = 0.0;
        MetricArray values{};
    };

    struct OnlineStats
    {
        std::uint64_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;

        void add(double value)
        {
            ++count;
            const double delta = value - mean;
            mean += delta / static_cast<double>(count);
            const double delta2 = value - mean;
            m2 += delta * delta2;
        }

        double stderr() const
        {
            if (count < 2)
            {
                return 0.0;
            }
            const double variance = m2 / static_cast<double>(count - 1);
            return std::sqrt(std::max(0.0, variance) /
                             static_cast<double>(count));
        }
    };

    struct PAccumulator
    {
        double p = std::numeric_limits<double>::quiet_NaN();
        std::array<OnlineStats, METRIC_COUNT> metrics{};
    };

    std::uint64_t parse_nonnegative_u64(const char *text, const char *name)
    {
        if (text == nullptr || *text == '\0' || text[0] == '-')
        {
            throw std::invalid_argument(
                std::string(name) + " must be a non-negative integer.");
        }

        char *end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0')
        {
            throw std::invalid_argument(
                std::string(name) + " must be a non-negative integer.");
        }
        return static_cast<std::uint64_t>(value);
    }

    std::size_t read_positive_size_env(const char *name, std::size_t fallback)
    {
        const char *text = std::getenv(name);
        if (text == nullptr || *text == '\0')
        {
            return fallback;
        }
        const std::uint64_t value = parse_nonnegative_u64(text, name);
        if (value == 0 || value > std::numeric_limits<std::size_t>::max())
        {
            throw std::invalid_argument(
                std::string(name) + " must be a positive integer.");
        }
        return static_cast<std::size_t>(value);
    }

    double read_nonnegative_double_env(const char *name, double fallback)
    {
        const char *text = std::getenv(name);
        if (text == nullptr || *text == '\0')
        {
            return fallback;
        }

        char *end = nullptr;
        errno = 0;
        const double value = std::strtod(text, &end);
        if (errno != 0 || end == text || *end != '\0' ||
            !std::isfinite(value) || value < 0.0)
        {
            throw std::invalid_argument(
                std::string(name) + " must be a finite non-negative number.");
        }
        return value;
    }

    std::string default_output_name(
        const mipt::io::DensityFileMetadata &metadata,
        std::uint64_t limit_per_p)
    {
        const std::uint64_t output_realizations =
            (limit_per_p == 0)
                ? metadata.realizations
                : std::min<std::uint64_t>(metadata.realizations, limit_per_p);

        std::ostringstream name;
        name << "expvals_n_" << metadata.total_qubits
             << "_real_" << mipt::util::compact_count(output_realizations) << ".csv";
        return name.str();
    }

    std::string format_duration(double seconds)
    {
        return mipt::util::format_duration(seconds, mipt::util::DurationStyle::Clock);
    }

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program
            << " [rho3.bin] [output.csv] [limit_per_p]\n\n"
            << "Defaults:\n"
            << "  input: rho3.bin\n"
            << "  output: expvals_n_<N>_real_<accepted-realizations>.csv\n"
            << "  limit_per_p: 0 (all records)\n\n"
            << "The input must contain trajectory-resolved three-mode 8x8 RDMs.\n"
            << "The output contains one aggregate row per p. Each trajectory is\n"
            << "first averaged over all stored subsystems; means and standard\n"
            << "errors are then calculated across trajectories.\n\n"
            << "Performance environment variables:\n"
            << "  EXPVALS_BATCH_RECORDS=<positive integer>  default 256\n"
            << "  EXPVALS_PROGRESS_SECONDS=<seconds>        default 2; 0 disables\n"
            << "  OMP_NUM_THREADS=<threads>                 OpenMP worker count\n"
            << "  OMP_PROC_BIND=close and OMP_PLACES=cores are often useful.\n";
    }

    TrajectorySummary summarize_record(
        const mipt::io::DensityRecord &record,
        std::uint32_t subsystem_count)
    {
        TrajectorySummary summary;
        summary.p_index = record.p_index;
        summary.p = record.p;

        for (std::uint32_t subsystem = 0; subsystem < subsystem_count; ++subsystem)
        {
            const std::size_t rho_offset =
                static_cast<std::size_t>(subsystem) * MATRIX_VALUE_COUNT;
            const auto values = mipt::analysis::compute_fermion_exp_vals_8x8(
                record.rho_ri.data() + rho_offset);

            summary.values[PURITY] += values.purity;
            summary.values[HOP_R1_SQ] +=
                0.5 * (values.hopping_squared[0] + values.hopping_squared[1]);
            summary.values[HOP_R2_SQ] += values.hopping_squared[2];
            summary.values[PAIR_R1_SQ] +=
                0.5 * (values.pairing_squared[0] + values.pairing_squared[1]);
            summary.values[PAIR_R2_SQ] += values.pairing_squared[2];
            summary.values[NUMBER_MEAN] += values.number_mean;
            summary.values[NUMBER_VAR] += values.number_variance;
            summary.values[PARITY_EXPECTATION] += values.parity_expectation;
            summary.values[PARITY_VAR] += values.parity_variance;
            summary.values[DENSITY_R1] +=
                0.5 * (values.connected_density[0] + values.connected_density[1]);
            summary.values[DENSITY_R2] += values.connected_density[2];
            summary.values[DENSITY_R1_SQ] +=
                0.5 * (values.connected_density_squared[0] +
                       values.connected_density_squared[1]);
            summary.values[DENSITY_R2_SQ] +=
                values.connected_density_squared[2];
            summary.values[WICK4] += values.wick4_residual_squared;
            summary.values[WICK6] += values.wick6_residual_squared;
        }

        const double inverse_subsystems = 1.0 / static_cast<double>(subsystem_count);
        for (double &value : summary.values)
        {
            value *= inverse_subsystems;
        }
        return summary;
    }

    void process_batch(
        const std::vector<mipt::io::DensityRecord> &batch,
        std::uint32_t subsystem_count,
        std::vector<PAccumulator> &accumulators,
        std::uint64_t &matrices_processed)
    {
        std::vector<TrajectorySummary> summaries(batch.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(batch.size());
             ++i)
        {
            summaries[static_cast<std::size_t>(i)] = summarize_record(
                batch[static_cast<std::size_t>(i)], subsystem_count);
        }

        for (const auto &summary : summaries)
        {
            if (summary.p_index >= accumulators.size())
            {
                throw std::runtime_error(
                    "Record p_index exceeds metadata resolution.");
            }
            auto &accumulator = accumulators[summary.p_index];
            if (std::isnan(accumulator.p))
            {
                accumulator.p = summary.p;
            }
            for (std::size_t metric = 0; metric < METRIC_COUNT; ++metric)
            {
                accumulator.metrics[metric].add(summary.values[metric]);
            }
        }

        matrices_processed +=
            static_cast<std::uint64_t>(batch.size()) * subsystem_count;
    }

    void write_csv(
        const std::string &output_path,
        const std::vector<PAccumulator> &accumulators,
        std::uint32_t subsystem_count)
    {
        std::ofstream output(output_path);
        if (!output)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }

        output << std::setprecision(17);
        output << "p,p_index,realizations,subsystems_per_realization";
        for (const char *name : METRIC_NAMES)
        {
            output << ',' << name << "_mean," << name << "_stderr";
        }
        output << '\n';

        for (std::size_t p_index = 0; p_index < accumulators.size(); ++p_index)
        {
            const auto &accumulator = accumulators[p_index];
            const std::uint64_t realizations = accumulator.metrics[0].count;
            if (realizations == 0)
            {
                continue;
            }

            output << accumulator.p << ',' << p_index << ',' << realizations << ','
                   << subsystem_count;
            for (const auto &metric : accumulator.metrics)
            {
                output << ',' << metric.mean << ',' << metric.stderr();
            }
            output << '\n';
        }

        output.flush();
        if (!output)
        {
            throw std::runtime_error("Failed while writing output CSV.");
        }
    }

    void print_progress(
        std::uint64_t records_scanned,
        std::uint64_t record_count,
        std::uint64_t records_accepted,
        std::uint64_t matrices_processed,
        const std::chrono::steady_clock::time_point &start)
    {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
                                   .count();
        const double matrix_rate =
            (seconds > 0.0) ? matrices_processed / seconds : 0.0;
        const double progress =
            (record_count > 0)
                ? std::clamp(static_cast<double>(records_scanned) /
                                 static_cast<double>(record_count),
                             0.0,
                             1.0)
                : 1.0;
        const double eta =
            (progress > 0.0) ? seconds * (1.0 - progress) / progress
                             : std::numeric_limits<double>::infinity();

        std::cout
            << '\r'
            << "Scanned " << records_scanned << '/' << record_count
            << " records; accepted " << records_accepted
            << "; processed " << matrices_processed << " RDMs; "
            << std::fixed << std::setprecision(1) << matrix_rate << " RDM/s; "
            << std::setprecision(1) << 100.0 * progress << "%; ETA "
            << format_duration(eta) << "     " << std::flush;
    }
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc > 1 &&
            (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
        {
            print_usage(argv[0]);
            return 0;
        }
        if (argc > 4)
        {
            throw std::invalid_argument(
                "Too many arguments. Use --help for usage.");
        }

        const std::string input_path = (argc > 1) ? argv[1] : "rho3.bin";
        mipt::io::DensityMatrixReader reader(input_path);
        const auto &metadata = reader.metadata();

        if (metadata.kept_qubits != MODE_COUNT ||
            metadata.matrix_dimension != MATRIX_DIMENSION)
        {
            throw std::runtime_error(
                "Input must contain three-mode 8x8 reduced density matrices.");
        }
        if (metadata.resolution == 0)
        {
            throw std::runtime_error("Input metadata has zero p resolution.");
        }

        const std::uint64_t limit_per_p =
            (argc > 3) ? parse_nonnegative_u64(argv[3], "limit_per_p") : 0;
        const std::string output_path =
            (argc > 2) ? argv[2] : default_output_name(metadata, limit_per_p);
        const std::size_t batch_records = read_positive_size_env(
            "EXPVALS_BATCH_RECORDS", DEFAULT_BATCH_RECORDS);
        const double progress_seconds = read_nonnegative_double_env(
            "EXPVALS_PROGRESS_SECONDS", DEFAULT_PROGRESS_SECONDS);

#ifdef _OPENMP
        const int thread_count = omp_get_max_threads();
#else
        const int thread_count = 1;
#endif

        const std::uint64_t per_p_target =
            (limit_per_p == 0)
                ? metadata.realizations
                : std::min<std::uint64_t>(metadata.realizations, limit_per_p);

        std::vector<std::uint64_t> accepted_per_p(metadata.resolution, 0);
        std::vector<PAccumulator> accumulators(metadata.resolution);
        std::vector<mipt::io::DensityRecord> batch;
        batch.reserve(batch_records);

        std::uint64_t records_scanned = 0;
        std::uint64_t records_accepted = 0;
        std::uint64_t records_skipped = 0;
        std::uint64_t matrices_processed = 0;
        std::size_t completed_p_values = 0;

        const auto start = std::chrono::steady_clock::now();
        auto last_progress = start;

        std::cout
            << "Reading " << input_path
            << " (N=" << metadata.total_qubits
            << ", records=" << metadata.record_count
            << ", subsystems/record=" << metadata.subsystem_count << ")\n"
            << "Output: " << output_path << '\n'
            << "OpenMP threads: " << thread_count
            << "; batch records: " << batch_records
            << "; limit_per_p: " << limit_per_p << '\n';

        auto flush_batch = [&]() {
            if (batch.empty())
            {
                return;
            }
            process_batch(
                batch,
                metadata.subsystem_count,
                accumulators,
                matrices_processed);
            batch.clear();
        };

        mipt::io::DensityRecord record;
        mipt::util::PauseSentinel pause_sentinel;

        while (reader.read_record(record))
        {
            // Safe checkpoint: between input records.
            pause_sentinel.wait(flush_batch);
            ++records_scanned;
            if (record.p_index >= accepted_per_p.size())
            {
                throw std::runtime_error(
                    "Record p_index exceeds metadata resolution.");
            }

            if (limit_per_p != 0 &&
                accepted_per_p[record.p_index] >= limit_per_p)
            {
                ++records_skipped;
            }
            else
            {
                ++accepted_per_p[record.p_index];
                ++records_accepted;
                if (per_p_target > 0 &&
                    accepted_per_p[record.p_index] == per_p_target)
                {
                    ++completed_p_values;
                }

                batch.push_back(std::move(record));
                record = mipt::io::DensityRecord{};
                if (batch.size() >= batch_records)
                {
                    flush_batch();
                }
            }

            const auto now = std::chrono::steady_clock::now();
            if (progress_seconds > 0.0 &&
                std::chrono::duration<double>(now - last_progress).count() >=
                    progress_seconds)
            {
                flush_batch();
                print_progress(
                    records_scanned,
                    metadata.record_count,
                    records_accepted,
                    matrices_processed,
                    start);
                last_progress = now;
            }

            if (limit_per_p != 0 &&
                completed_p_values == metadata.resolution)
            {
                break;
            }
        }

        flush_batch();
        if (progress_seconds > 0.0)
        {
            print_progress(
                records_scanned,
                records_scanned,
                records_accepted,
                matrices_processed,
                start);
            std::cout << '\n';
        }

        write_csv(output_path, accumulators, metadata.subsystem_count);

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
                                   .count();
        const double rate =
            (elapsed > 0.0) ? matrices_processed / elapsed : 0.0;

        std::cout
            << "Wrote " << output_path << " with one row per accepted p value.\n"
            << "Records accepted: " << records_accepted
            << "; skipped by limit_per_p: " << records_skipped
            << "; RDMs processed: " << matrices_processed << '\n'
            << "Elapsed: " << format_duration(elapsed)
            << "; average throughput: " << std::fixed << std::setprecision(1)
            << rate << " RDM/s\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "exp_vals.exe error: " << error.what() << '\n';
        return 1;
    }
}

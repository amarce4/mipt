#pragma once

#include "mipt/tmi/compute.hpp"
#include "mipt/util/pause.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace mipt::tmi
{
inline const char *compile_target_name()
{
    return mipt::backend::compiled_cudaq_backend();
}

inline const char *cuda_rho_build_name()
{
#ifdef MIPT_ENABLE_CUDA_RHO
    return "enabled";
#else
    return "disabled";
#endif
}

inline const char *state_precision_name(cudaq::SimulationState::precision precision)
{
    return precision == cudaq::SimulationState::precision::fp64 ? "fp64" : "fp32";
}

inline std::string format_duration(double seconds)
{
    return util::format_duration(seconds, util::DurationStyle::Compact);
}

inline void validate_sim_args(int n, int periods, int realizations, int res, double p_min, double p_max)
{
    if (n < 4 || (n % 4) != 0)
    {
        throw std::invalid_argument("Online TMI requires n >= 4 and n divisible by 4.");
    }
    if (periods <= 0)
    {
        throw std::invalid_argument("periods must be positive.");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (res <= 0)
    {
        throw std::invalid_argument("resolution must be positive.");
    }
    if (!(0.0 <= p_min && p_min <= 1.0 && 0.0 <= p_max && p_max <= 1.0))
    {
        throw std::invalid_argument("p_min and p_max must lie in [0, 1].");
    }
}

void ensure_output_parent_directory(const std::string &output_path);

inline void run_1d_sim_tmi(int n,
                    int periods,
                    int realizations,
                    int res,
                    double p_min,
                    double p_max,
                    CircuitType circuit_mode,
                    bool all_cycles,
                    const std::string &output_path)
{
    validate_sim_args(n, periods, realizations, res, p_min, p_max);
    validate_circuit_site_count(circuit_mode, n);
    // Resolve before anything is printed or opened: an unusable
    // SIM_TMI_ENTROPY_ORDER throws here rather than partway through the banner.
    const EntropyOrder entropy_order = tmi_entropy_order();

    const std::uint32_t block_qubits = static_cast<std::uint32_t>(n / 4);
    const int cycle_count = tmi_cycle_count(n, all_cycles);
    const bool fermion_trace = uses_fermionic_trace(circuit_mode);
    const TmiPlans plans = make_tmi_plans(block_qubits,
                                         fermion_trace ? TraceMode::Fermion : TraceMode::Qubit);
    const std::vector<double> ps = linspace(p_min, p_max, res);
    const std::uint64_t total = static_cast<std::uint64_t>(res) *
                                static_cast<std::uint64_t>(realizations) *
                                static_cast<std::uint64_t>(cycle_count);

    ensure_output_parent_directory(output_path);

    std::ofstream csv(output_path, std::ios::binary | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not create output CSV: " + output_path);
    }
    csv << "p,tmi\n";
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing output CSV header.");
    }

    const std::uint64_t flush_rows = csv_flush_row_interval(n, cycle_count);
    util::PauseSentinel pause_sentinel("SIM_TMI_PAUSE_FILE");
    mipt::backend::print_backend_banner_once("sim_tmi.exe");

    std::cerr << "Online TMI simulation\n"
              << "n=" << n
              << ", periods=" << periods
              << ", realizations=" << realizations
              << ", resolution=" << res
              << ", p_min=" << p_min
              << ", p_max=" << p_max
              << ", block_qubits=" << block_qubits
              << ", all_cycles=" << (all_cycles ? 1 : 0)
              << ", tmi_cycles_per_circuit=" << cycle_count
              << ", mode=" << circuit_type_name(circuit_mode)
              << ", entropy_order=" << entropy_order_name(entropy_order)
              << ", output=" << output_path << '\n'
              << "build_target=" << compile_target_name()
              << ", cuda_rho=" << cuda_rho_build_name()
              << ", device_small_rdm=" << (device_small_rdm_enabled() ? 1 : 0) << '\n'
              << "entropy_backend=" << lapack_runtime::status() << '\n'
              << "tmi_method=selected after first state; fast path requires zheevd plus either host SVD or enabled CUDA device half-SVD" << '\n'
              << "cuda_svd_min_kept_modes=" << cuda_svd_min_kept_modes()
              << ", small_n_cuda_svd_policy="
              << (force_cuda_svd_for_small_n() ? "forced-on" : "cpu-for-n8-n12")
              << ", csv_flush_rows=" << flush_rows
              << '\n';
    std::cerr << pause_sentinel.describe() << '\n';

    TmiWorkspace workspace;
    std::string line;
    line.reserve(128);
    std::string csv_buffer;
    csv_buffer.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(flush_rows, 4096)) * 48u);
    std::uint64_t buffered_rows = 0;
    auto flush_csv_buffer = [&]() {
        if (csv_buffer.empty())
        {
            return;
        }
        csv.write(csv_buffer.data(), static_cast<std::streamsize>(csv_buffer.size()));
        csv_buffer.clear();
        buffered_rows = 0;
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while writing output CSV.");
        }
    };

    CircuitWorkspace1D circuit_workspace;
    circuit_workspace.reserve(periods);

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    std::uint64_t processed = 0;
    std::uint64_t last_processed = 0;
    bool printed_state_backend = false;

    auto wait_if_host_paused = [&]() {
        const double duration = pause_sentinel.wait(flush_csv_buffer);
        // Keep the progress line from reporting a burst after a long pause.
        last_report += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(duration));
    };

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        for (int r = 0; r < realizations; ++r)
        {
            wait_if_host_paused();

            auto state = circuit_workspace.simulate(
                n, periods, p, circuit_mode, "sim_tmi");

            const bool use_fast_state_tmi = fast_state_tmi_available(state, n, block_qubits);
            if (!printed_state_backend)
            {
                const bool device_tensor = cudaq_state_has_device_tensor(state);
                const bool host_svd_available = host_svd_available_for_state(state);
                const std::size_t half_cut_kept_modes = 2u * static_cast<std::size_t>(block_qubits);
                const bool small_n_cuda_svd_disabled = cuda_svd_disabled_for_small_n(n);
                const bool cuda_half_svd_enabled = device_tensor &&
                                                   should_try_cuda_svd_for_problem(n, half_cut_kept_modes);
                const char *half_cut_backend = cuda_half_svd_enabled
                                                   ? "CUDA/cuSOLVER"
                                                   : (small_n_cuda_svd_disabled
                                                          ? "host LAPACK (N=8/12 CPU override)"
                                                          : (host_svd_available ? "host LAPACK" : "legacy RDM fallback"));
                std::cerr << "state_backend=" << (state.is_on_gpu() ? "gpu" : "host")
                          << ", precision=" << state_precision_name(state.get_precision())
                          << ", device_tensor=" << (device_tensor ? 1 : 0)
                          << ", host_svd_available=" << (host_svd_available ? 1 : 0)
                          << ", small_n_cuda_svd_disabled=" << (small_n_cuda_svd_disabled ? 1 : 0)
                          << ", cuda_half_svd_enabled=" << (cuda_half_svd_enabled ? 1 : 0)
                          << ", device_small_rdm=" << (device_small_rdm_enabled() ? 1 : 0)
                          << ", half_cut_backend=" << half_cut_backend
                          << ", fast_state_tmi=" << (use_fast_state_tmi ? 1 : 0)
                          << '\n';
                printed_state_backend = true;
            }

            std::vector<double> tmi_values;
            if (use_fast_state_tmi)
            {
                try
                {
                    tmi_values = tmi_values_from_cudaq_state_fast(state,
                                                                  n,
                                                                  block_qubits,
                                                                  cycle_count,
                                                                  fermion_trace,
                                                                  workspace);
                }
                catch (const std::bad_alloc &)
                {
                    throw std::runtime_error(
                        "Out of host memory while building a Schmidt matrix for fast TMI. "
                        "For N=24 and especially N=28, the exact half-cut SVD still requires a large "
                        "host matrix; ensure enough RAM/swap or reduce concurrent memory pressure.");
                }
            }
            else
            {
                // Compatibility path for systems without a usable fast state-vector path.
                // This is exact but much slower because it materializes pair RDMs.
                tmi_values = tmi_values_from_cudaq_state_payload(state,
                                                                 n,
                                                                 block_qubits,
                                                                 cycle_count,
                                                                 fermion_trace,
                                                                 plans,
                                                                 workspace);
            }

            for (double tmi : tmi_values)
            {
                line.clear();
                append_double(line, p);
                line.push_back(',');
                append_double(line, tmi);
                line.push_back('\n');

                csv_buffer.append(line);
                ++buffered_rows;

                ++processed;
                const auto now = std::chrono::steady_clock::now();
                const double since_last = std::chrono::duration<double>(now - last_report).count();
                if (since_last >= 1.0 || processed == total)
                {
                    const double wall_elapsed = std::chrono::duration<double>(now - start).count();
                    const double active_elapsed = std::max(1.0e-12, wall_elapsed - pause_sentinel.paused_seconds());
                    const double avg = static_cast<double>(processed) / active_elapsed;
                    const double recent = static_cast<double>(processed - last_processed) / std::max(1.0e-12, since_last);
                    const double eta = avg > 0.0 ? static_cast<double>(total - processed) / avg : 0.0;
                    std::cerr << '\r'
                              << "processed TMI " << processed << '/' << total
                              << " | avg " << std::fixed << std::setprecision(2) << avg << "/s"
                              << " | recent " << std::fixed << std::setprecision(2) << recent << "/s"
                              << " | ETA " << format_duration(eta)
                              << "          " << std::flush;
                    last_report = now;
                    last_processed = processed;
                }
            }

            if (buffered_rows >= flush_rows || processed == total)
            {
                flush_csv_buffer();
            }
        }
    }

    flush_csv_buffer();
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while finalizing output CSV.");
    }
    const auto finish = std::chrono::steady_clock::now();
    const double wall_total = std::chrono::duration<double>(finish - start).count();
    const double active_total = std::max(1.0e-12, wall_total - pause_sentinel.paused_seconds());
    std::cerr << '\n'
              << "Completed " << processed << " TMI row(s) in "
              << format_duration(active_total) << " active time";
    if (pause_sentinel.paused_seconds() > 0.0)
    {
        std::cerr << " plus " << format_duration(pause_sentinel.paused_seconds()) << " paused by host";
    }
    std::cerr << ". Overall active throughput "
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(processed) / active_total) << "/s.\n";
    if (workspace.cuda_svd_disabled)
    {
        std::cerr << "Warning: cuSOLVER half-cut SVD was disabled after status "
                  << workspace.cuda_svd_failure_status
                  << "; using host LAPACK SVD for half cuts.\n";
    }
}

inline bool is_bool01_arg(std::string_view value)
{
    return value == "0" || value == "1";
}

inline bool parse_bool01_arg(std::string_view value, const char *name)
{
    if (value == "0")
    {
        return false;
    }
    if (value == "1")
    {
        return true;
    }
    throw std::invalid_argument(std::string(name) + " must be 0 or 1.");
}

inline std::string format_number_for_filename(double value)
{
    if (std::abs(value) < 1.0e-15)
    {
        value = 0.0;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(12) << value;
    std::string text = out.str();

    const std::size_t dot = text.find('.');
    if (dot != std::string::npos)
    {
        while (!text.empty() && text.back() == '0')
        {
            text.pop_back();
        }
        if (!text.empty() && text.back() == '.')
        {
            text.pop_back();
        }
    }

    if (text == "-0")
    {
        text = "0";
    }
    return text;
}

inline std::string tmi_output_tag(CircuitType circuit_mode, bool all_cycles)
{
    std::string tag(circuit_type_tag(circuit_mode));
    if (all_cycles) tag += 'c';
    // I_3 built from S_2 is a different universal number from I_3 built from
    // S_1, so the two must never end up in one glob.  The suffix lands in both
    // the directory and the file stem; order-1 paths are unchanged.
    if (tmi_entropy_order() == EntropyOrder::Renyi2) tag += "_s2";
    return tag;
}

inline std::string default_tmi_output_path(int n,
                                    int realizations,
                                    int res,
                                    double p_min,
                                    double p_max,
                                    CircuitType circuit_mode,
                                    bool all_cycles)
{
    const std::string tag = tmi_output_tag(circuit_mode, all_cycles);

    std::ostringstream filename;
    filename << "tmi_" << tag
             << "_n_" << n
             << "_real_" << realizations
             << '_' << format_number_for_filename(p_min)
             << '_' << format_number_for_filename(p_max)
             << "_res_" << res
             << ".csv";

    return std::string("csv/tmi/") + tag + "/" + filename.str();
}

inline void make_directory_if_needed(const std::string &path)
{
    if (path.empty())
    {
        return;
    }

    if (::mkdir(path.c_str(), 0755) == 0)
    {
        return;
    }

    if (errno == EEXIST)
    {
        return;
    }

    throw std::runtime_error("Could not create output directory: " + path + " (" + std::strerror(errno) + ")");
}

inline void ensure_output_parent_directory(const std::string &output_path)
{
    const std::size_t slash = output_path.find_last_of("/\\");
    if (slash == std::string::npos)
    {
        return;
    }

    const std::string parent = output_path.substr(0, slash);
    if (parent.empty())
    {
        return;
    }

    std::string current;
    std::size_t pos = 0;
    if (parent[0] == '/')
    {
        current = "/";
        pos = 1;
    }

    while (pos < parent.size())
    {
        const std::size_t next = parent.find_first_of("/\\", pos);
        const std::string part = parent.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (!part.empty())
        {
            if (!current.empty() && current.back() != '/')
            {
                current += '/';
            }
            current += part;
            make_directory_if_needed(current);
        }
        if (next == std::string::npos)
        {
            break;
        }
        pos = next + 1;
    }
}

inline void print_usage(const char *argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " [n = 10] [periods = 10] [realizations = 10]"
           " [resolution = 5] [p_min = 0.0] [p_max = 1.0]"
           " [circ_type = 0] [all_cycles = 0] [output.csv = auto]\n\n"
        << "Arguments:\n"
        << "  circ_type:\n";
    print_circuit_type_help(std::cerr, "    ");
    std::cerr
        << "  all_cycles = 0 computes the original offset only.\n"
        << "  all_cycles = 1 computes N/4 cyclic offsets; e.g. N=8 gives [0,1]|[2,3]|[4,5]|[6,7] and [1,2]|[3,4]|[5,6]|[7,0].\n"
        << "  For backward compatibility, output.csv may also be supplied before all_cycles.\n"
        << "  If output.csv is omitted, the default path is:\n"
        << "    csv/tmi/<tag>/tmi_<tag>_n_<n>_real_<realizations>_<p_min>_<p_max>_res_<resolution>.csv\n"
        << "  Tags are mms, haar, rppu, rfgs, and qrppu; all_cycles=1 appends c,\n"
        << "  and SIM_TMI_ENTROPY_ORDER=2 appends _s2.\n\n"
        << "Output CSV columns:\n"
        << "  p,tmi\n\n"
        << "Entropy order:\n"
        << "  SIM_TMI_ENTROPY_ORDER=1 (default) builds I_3 from the von Neumann entropy.\n"
        << "  SIM_TMI_ENTROPY_ORDER=2 builds it from the Renyi-2 entropy -log2 Tr(rho^2).\n"
        << "  Order 2 needs no spectrum, so it skips the half-cut eigensolve entirely --\n"
        << "  about 98% of a TMI evaluation at N=24, and 6.2 GB of cuSOLVER workspace at\n"
        << "  the 16384-wide half cut of N=28.  I_3 at criticality is Renyi-index\n"
        << "  dependent, so order-2 output must not be pooled with order-1 output; p_c\n"
        << "  and nu are unaffected.  Output paths carry an _s2 tag to keep them apart.\n\n"
        << "Host pause control:\n"
        << "  By default, creating PAUSE_MIPT in the process working directory pauses at the next safe checkpoint.\n"
        << "  Removing PAUSE_MIPT resumes the run. Set SIM_TMI_PAUSE_FILE=/path/to/file to use another sentinel,\n"
        << "  or SIM_TMI_PAUSE_FILE=0 to disable this mechanism. Progress rates exclude paused time.\n";
}

} // namespace mipt::tmi

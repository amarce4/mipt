#include "mipt/backend.hpp"
#include "mipt/circuit.hpp"
#include "mipt/density.hpp"
#include "mipt/native_mps.hpp"
#include "mipt/rdm.hpp"
#include "mipt/util/pause.hpp"
#include <algorithm>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace mipt;

class CircuitRateTracker
{
  public:
    void observe(int realization, double seconds)
    {
        if (realization == 0)
        {
            seconds_per_circuit_ = seconds;
            steady_samples_ = 0;
            return;
        }
        seconds_per_circuit_ =
            (seconds_per_circuit_ * static_cast<double>(steady_samples_) + seconds) /
            static_cast<double>(steady_samples_ + 1);
        ++steady_samples_;
    }

    double rate() const noexcept
    {
        return seconds_per_circuit_ > 0.0 ? 1.0 / seconds_per_circuit_ : 0.0;
    }

    double eta_seconds(int completed, int total) const noexcept
    {
        return static_cast<double>(std::max(0, total - completed)) *
               seconds_per_circuit_;
    }

  private:
    double seconds_per_circuit_ = 0.0;
    int steady_samples_ = 0;
};

void print_first_trajectory_timing(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point frontend_done,
    std::chrono::steady_clock::time_point state_done,
    std::chrono::steady_clock::time_point observable_done,
    std::chrono::steady_clock::time_point save_done)
{
    std::cout << "Layer, QC, DM, Save times: "
              << std::chrono::duration<double>(frontend_done - start).count() << "s, "
              << std::chrono::duration<double>(state_done - frontend_done).count() << "s, "
              << std::chrono::duration<double>(observable_done - state_done).count() << "s, "
              << std::chrono::duration<double>(save_done - observable_done).count() << "s\n";
}

void print_progress(const CircuitRateTracker &tracker, int completed, int total)
{
    std::cout << "\rAverage circuits/second: "
              << std::round(tracker.rate() * 100.0) / 100.0
              << ", Realisations ETA: "
              << static_cast<int>(tracker.eta_seconds(completed, total))
              << "s               " << std::flush;
}

void run_1d(int n, int periods, int realizations, int res, double p_min, double p_max, CircuitType circ_type)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires n >= 3.");
    }
    validate_circuit_site_count(circ_type, n);

    const bool fermion_trace = uses_fermionic_trace(circ_type);
    const std::vector<Subsystem> subsystems = periodic_ring_three_site_subsystems(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info.csv");

    infofile << "n,periods,realizations,resolution,p_min,p_max,circ_type,circuit_name,fermion_trace,cudaq_backend\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "," << static_cast<int>(circ_type) << ","
             << circuit_type_name(circ_type) << "," << (fermion_trace ? 1 : 0) << ","
             << mipt::backend::compiled_cudaq_backend() << "\n";

    mipt::io::DensityMatrixWriter rho3_file(
        "rho3.bin",
        density_metadata(1, n, periods, realizations, res,
                         p_min, p_max, n, 1, subsystems));

    CircuitWorkspace1D circuit_workspace;
    circuit_workspace.reserve(periods);
    std::mt19937 native_mps_rng(std::random_device{}());
    std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);

    mipt::util::PauseSentinel pause_sentinel;

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        CircuitRateTracker rate_tracker;

        for (int r = 0; r < realizations; ++r)
        {
            // Safe checkpoint: the previous trajectory is already written.
            pause_sentinel.wait([&]() { rho3_file.flush(); });
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            std::fill(rho3_ri.begin(), rho3_ri.end(), 0.0);

            if (mipt::native_mps::enabled())
            {
                if (r == 0)
                {
                    lap1 = std::chrono::steady_clock::now();
                    mipt::native_mps::log("MIPT_NATIVE_MPS=1: using native CPU TEBD/MPS path; CUDA-Q get_state is bypassed for this trajectory.");
                }

                mipt::native_mps::simulate_1d_rho3(
                    n, periods, p, circ_type, subsystems, rho3_ri.data(), native_mps_rng);

                if (r == 0)
                {
                    lap2 = std::chrono::steady_clock::now();
                }
            }
            else
            {
                CircuitBuildTiming build_timing;
                auto state = circuit_workspace.simulate(
                    n, periods, p, circ_type, {},
                    r == 0 ? &build_timing : nullptr);

                if (r == 0)
                {
                    lap1 = build_timing.frontend_done;
                    lap2 = std::chrono::steady_clock::now();
                }

                cudaq_three_site_density_matrices(
                    state, n, subsystems, rho3_ri.data(), fermion_trace);
            }

            if (r == 0)
            {
                lap3 = std::chrono::steady_clock::now();
            }

            rho3_file.write_record(
                static_cast<std::uint32_t>(p_index),
                static_cast<std::uint32_t>(r),
                p,
                rho3_ri.data(),
                rho3_ri.size());

            if (r == 0)
            {
                lap4 = std::chrono::steady_clock::now();

                print_first_trajectory_timing(start, lap1, lap2, lap3, lap4);
            }

            const double circuit_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - circ_time_start).count();
            rate_tracker.observe(r, circuit_seconds);
            print_progress(rate_tracker, r + 1, realizations);
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved " << subsystems.size()
              << " cyclic three-qubit reduced density matrices per trajectory to rho3.bin for circ_type "
              << static_cast<int>(circ_type) << " (" << circuit_type_name(circ_type) << ")"
              << (fermion_trace ? " using fermionic partial traces.\n" : ".\n");
}




void run_2d(int x, int y, int periods, int realizations, int res, double p_min, double p_max)
{
    const int n = x * y;
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires x*y >= 3.");
    }

    // The requested distance-one window convention is defined for the 1D ring.
    // Preserve the prior 2D output behaviour as one terminal three-qubit RDM.
    const std::vector<Subsystem> subsystems = three_site_subsystems_2d(x, y);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info2.csv");

    // Keep the existing info2.csv format unchanged.
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "\n";

    mipt::io::DensityMatrixWriter rho3_file(
        "rho3_2d.bin",
        density_metadata(2, n, periods, realizations, res,
                         p_min, p_max, x, y, subsystems));

    std::mt19937 mms_rng(std::random_device{}());
    std::vector<MmsLayer> mms_layers;
    mms_layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);

    mipt::util::PauseSentinel pause_sentinel;

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        CircuitRateTracker rate_tracker;

        for (int r = 0; r < realizations; ++r)
        {
            // Safe checkpoint: the previous trajectory is already written.
            pause_sentinel.wait([&]() { rho3_file.flush(); });
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            build_mms_layers(mms_layers, n, periods, p, mms_rng);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            if (mipt::backend::verbose_enabled())
            {
                mipt::backend::verbose_log("get_state start: MMS2D n=" + std::to_string(n) + " layers=" + std::to_string(mms_layers.size()));
            }
            const auto t_get_state = std::chrono::steady_clock::now();
            auto state = cudaq::get_state(MmsKernel2D{}, x, y, mms_layers, true);
            if (mipt::backend::verbose_enabled())
            {
                mipt::backend::verbose_log("get_state done: MMS2D elapsed_s=" + std::to_string(mipt::backend::seconds_since(t_get_state)));
            }

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::fill(rho3_ri.begin(), rho3_ri.end(), 0.0);
            cudaq_three_site_density_matrices(
                state, n, subsystems, rho3_ri.data());

            if (r == 0)
            {
                lap3 = std::chrono::steady_clock::now();
            }

            rho3_file.write_record(
                static_cast<std::uint32_t>(p_index),
                static_cast<std::uint32_t>(r),
                p,
                rho3_ri.data(),
                rho3_ri.size());

            if (r == 0)
            {
                lap4 = std::chrono::steady_clock::now();

                print_first_trajectory_timing(start, lap1, lap2, lap3, lap4);
            }

            const double circuit_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - circ_time_start).count();
            rate_tracker.observe(r, circuit_seconds);
            print_progress(rate_tracker, r + 1, realizations);
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved reduced density matrices to rho3_2d.bin.\n";
}


int run_cli(int argc, char *argv[])
{
    mipt::backend::print_backend_banner_once("mipt.exe");
    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        std::cout << "Usage: " << argv[0]
                  << " [d = 1] [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [circ_type = 0]\n";
        std::cout << "Description:\n";
        std::cout << "  d = dimensionality (1 or 2)\n";
        std::cout << "  n = number of qubits per dimension; for d=2 this means an n x n lattice\n";
        std::cout << "  periods = number of brickwork periods (1D period = even layer + odd layer)\n";
        std::cout << "  realizations = number of realizations per p value\n";
        std::cout << "  resolution = number of p grid points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        std::cout << "  circ_type:\n";
        print_circuit_type_help(std::cout, "    ");
        std::cout << "  For d=2, only circ_type=0 is supported.\n";
        return 0;
    }

    if (argc > 9)
    {
        std::cerr << "Too many arguments. Use --help for usage.\n";
        return 2;
    }

    int d = (argc > 1) ? std::stoi(argv[1]) : 1;
    int n = (argc > 2) ? std::stoi(argv[2]) : ((d == 1) ? 10 : 4);
    int periods = (argc > 3) ? std::stoi(argv[3]) : ((d == 1) ? 10 : 32);
    int realizations = (argc > 4) ? std::stoi(argv[4]) : 10;
    int res = (argc > 5) ? std::stoi(argv[5]) : 5;
    double p_min = (argc > 6) ? std::stod(argv[6]) : 0.0;
    double p_max = (argc > 7) ? std::stod(argv[7]) : 1.0;
    CircuitType circ_type = parse_circuit_type((argc > 8) ? std::stoi(argv[8]) : 0);

    if (n <= 0)
    {
        throw std::invalid_argument("n must be positive.");
    }
    if (d != 1 && d != 2)
    {
        throw std::invalid_argument("d must be 1 or 2.");
    }
    if (res <= 0)
    {
        throw std::invalid_argument("resolution must be positive.");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (periods < 0)
    {
        throw std::invalid_argument("periods must be non-negative.");
    }
    if (p_min < 0.0 || p_max > 1.0 || p_min > p_max)
    {
        throw std::invalid_argument("Require 0 <= p_min <= p_max <= 1.");
    }

    auto start = std::chrono::steady_clock::now();

    if (d == 1)
    {
        std::cout << "Running 1D " << n << "-qubit circ_type "
                  << static_cast<int>(circ_type) << " (" << circuit_type_name(circ_type)
                  << ") simulation of " << realizations * res << " circuits.\n";
        run_1d(n, periods, realizations, res, p_min, p_max, circ_type);
    }
    else
    {
        if (circ_type != CircuitType::MMS)
        {
            throw std::invalid_argument("For d=2, only circ_type=0 (MMS gate set) is supported.");
        }
        std::cout << "Running 2D MMS " << n << "x" << n << " = " << n * n
                  << "-qubit simulation of " << realizations * res << " circuits.\n";
        run_2d(n, n, periods, realizations, res, p_min, p_max);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        return run_cli(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "mipt.exe error: " << error.what() << '\n';
        return 1;
    }
}

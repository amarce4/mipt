// Unified 1-, 2-, 3-, and 4-reference probe simulator.
//
// This file only parses, dispatches, and reports. The protocols live in
// include/mipt/probed/:
//
//   config.hpp      argument parsing into one ProbeRunConfig
//   geometry.hpp    where the references attach, and the mode-5 receiver grid
//   kernels.hpp     state preparation and reference readout
//   front.hpp       mode-5 receiver-block readout
//   runner.hpp      the trajectory loop for all six modes
//   sdp.hpp         deferred GMN/fGMN batches for three probes
//   output.hpp      per-mode CSV schemas
//   checkpoint.hpp  per-p subprocess isolation and resume
//   progress.hpp    the progress line
//
// Run with --help for the protocols and environment controls.

#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/probed/checkpoint.hpp"
#include "mipt/probed/config.hpp"
#include "mipt/probed/output.hpp"
#include "mipt/probed/progress.hpp"
#include "mipt/probed/runner.hpp"
#include "mipt/util/text.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>

namespace
{
using namespace mipt;
using namespace mipt::probed;

int run_cli(int argc, char *argv[])
{
    const bool internal_child = env::boolean("MIPT_PROBED_INTERNAL_CHILD", false);
    if (!internal_child)
    {
        backend::print_backend_banner_once("mipt_probed.exe");
    }
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))
    {
        print_help(argv[0]);
        return 0;
    }

    const ProbeRunConfig config = parse_probe_arguments(argc, argv);
    if (!internal_child)
    {
        std::cout << "Unified probed MIPT simulation\n" << describe_run(config) << '\n' << std::flush;
    }

    // A multi-point p scan is executed one child process per point, which is
    // what provides checkpointing and resume.
    if (config.mode != 0 && config.p_values.size() > 1 && env::boolean("MIPT_PROBED_ISOLATE_P", true))
    {
        return run_isolated_p_scan(argv[0], config.probes, config.n, config.realizations, config.type, config.mode,
                                   config.p_values, config.times, config.output);
    }

    const auto start = std::chrono::steady_clock::now();
    ProgressReporter progress(config.total_trajectories);
    const auto rows = simulate(config, progress);
    progress.finish();

    write_csv(config.output, config.probes, config.mode, config.n, config.realizations, config.type, rows,
              config.delta_x, config.delta_t, config.t_eq, config.triangle_balance_cutoff);

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!internal_child)
    {
        std::cout << "Saved " << rows.size() << " rows to " << config.output
                  << ". Elapsed time: " << util::format_duration(elapsed) << '\n';
    }
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
        std::cerr << "mipt_probed.exe error: " << error.what() << '\n';
        return 1;
    }
}

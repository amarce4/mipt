// Online distance scaling of two- and three-party entanglement measures.

#include "mipt/dist_gap_audit.hpp"
#include "mipt/dist_replay.hpp"
#include "mipt/dist_scaling.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    // MOSEK Fusion dies on a signal rather than throwing, so this has to be in
    // place before any solve. `run` refines the context once the run's settings
    // are known; see util/crash_report.hpp.
    mipt::util::crash::install("dist_scaling.exe");
    mipt::util::crash::set_context("run: dist_scaling.exe (still parsing arguments)");
    try
    {
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 ||
             std::strcmp(argv[1], "-h") == 0))
        {
            mipt::dist::print_usage(argv[0]);
            return 0;
        }
        // Offline re-analysis of a finished run: no circuit, no GPU, no
        // simulation. It reads the named files and writes new ones beside
        // them, and never modifies its inputs.
        if (argc > 1 && std::strcmp(argv[1], "--gap-audit") == 0)
        {
            if (argc < 3)
            {
                std::cerr << "--gap-audit needs the individual-outcome CSV:\n"
                          << "  " << argv[0]
                          << " --gap-audit <outcome.csv> [rho3.bin] [pair_records.bin]\n";
                return 2;
            }
            mipt::util::crash::set_context("run: dist_scaling.exe --gap-audit");
            // MOSEK reads its tolerance when the Fusion model is built, so
            // the audit's tolerance has to be in the environment before the
            // first solve rather than passed to it.
            const std::string audit_tol =
                mipt::env::text("MIPT_DIST_AUDIT_MOSEK_TOL", "1e-8");
            ::setenv("GMN_MOSEK_TOL", audit_tol.c_str(), 1);
            const mipt::dist::gap::FgmnSolver solver = [](const double *rho) {
                FgmnCertificate cert{};
                compute_fgmn_mosek_8x8_certified_cpp(rho, &cert);
                mipt::dist::gap::SolveOutcome outcome;
                outcome.value = cert.value;
                outcome.lower_bound = cert.lower_bound;
                outcome.upper_bound = cert.upper_bound;
                outcome.primal_residual = cert.primal_residual;
                outcome.dual_residual = cert.dual_residual;
                outcome.solver_gap = cert.solver_gap;
                outcome.status = cert.status;
                return outcome;
            };
            return mipt::dist::gap::audit::run_audit(
                argv[2], argc > 3 ? argv[3] : "", argc > 4 ? argv[4] : "", std::cout, solver);
        }
        if (argc > 9)
        {
            std::cerr << "Too many arguments. Use --help for usage.\n";
            return 2;
        }

        mipt::dist::RunConfig config;
        config.k = (argc > 1) ? std::stoi(argv[1]) : 2;
        config.n = (argc > 2) ? std::stoi(argv[2]) : 10;
        config.periods = (argc > 3) ? std::stoi(argv[3]) : 10;
        config.p = (argc > 4) ? std::stod(argv[4]) : 0.17;
        config.realizations = (argc > 5) ? std::stoi(argv[5]) : 10;
        config.type = mipt::parse_circuit_type((argc > 6) ? std::stoi(argv[6]) : 0);
        config.triangle_balance_cutoff =
            (argc > 8) ? std::stod(argv[8])
                       : mipt::env::real("MIPT_DIST_TRIANGLE_BALANCE", 0.5, 0.0, 1.0);
        // An explicit output path is taken as given. Left empty, it is filled in
        // by `run` once the arguments have been validated -- the default depends
        // on all of them, and k=0 needs two paths rather than one.
        if (argc > 7 && argv[7][0] != '\0')
        {
            config.output_path = std::string(argv[7]);
        }

        // Everything that is not a positional argument -- thresholds,
        // connectivity, record detail and stride, the master seed -- comes from
        // the environment and is resolved in one place so that k=0's two
        // configs cannot drift apart. `--help` is the authoritative list.
        mipt::dist::apply_environment(config);

        // A replay is an offline re-analysis, not a run: it regenerates the
        // named trajectories from the master seed and compares two precisions
        // on the same forced measurement record. It writes only its own file.
        const std::string replay_ids = mipt::env::text("MIPT_DIST_REPLAY_IDS", "");
        if (!replay_ids.empty())
        {
            if (config.type != mipt::CircuitType::FermionRPPU)
            {
                std::cerr << "error: MIPT_DIST_REPLAY_IDS needs circ_type 2 (RPPU); the replay "
                             "regenerates RPPU layers.\n";
                return 2;
            }
            mipt::util::crash::set_context("run: dist_scaling.exe (high-precision replay)");
            mipt::dist::replay::ReplaySettings settings;
            settings.ids_path = replay_ids;
            settings.third = static_cast<int>(
                mipt::env::integer("MIPT_DIST_REPLAY_THIRD", -1, -1, 4096));
            return mipt::dist::replay::run_replay(config, settings, std::cout);
        }

        mipt::dist::run(config);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\nerror: " << error.what() << '\n';
        return 1;
    }
}

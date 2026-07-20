#pragma once

#include "mipt/env.hpp"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>

#ifndef MIPT_CUDAQ_BACKEND_NAME
#define MIPT_CUDAQ_BACKEND_NAME "unknown"
#endif

#ifndef MIPT_CUDAQ_PRECISION
#define MIPT_CUDAQ_PRECISION 0
#endif

namespace mipt::backend
{
    inline long debug_prefix_layers()
    {
        return env::integer("MIPT_DEBUG_PREFIX_LAYERS", 0, 0, 1000000L);
    }


    inline bool verbose_enabled()
    {
        return env::boolean("MIPT_VERBOSE", false) ||
               env::boolean("MIPT_MPS_VERBOSE", false) ||
               env::boolean("SIM_TMI_VERBOSE", false);
    }

    inline long mps_amplitude_batch_size()
    {
        return env::integer("MIPT_MPS_AMPLITUDE_BATCH", 4096, 1, 1000000L);
    }

    inline double seconds_since(std::chrono::steady_clock::time_point start)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    inline void verbose_log(const std::string &message)
    {
        if (verbose_enabled())
        {
            std::cerr << "[mipt-debug] " << message << '\n';
        }
    }

    inline int amplitude_reverse_bits()
    {
        return env::boolean("MIPT_STATE_AMPLITUDE_REVERSE_BITS", false) ? 1 : 0;
    }

    inline long mps_exact_observable_max_qubits()
    {
        return env::integer("MIPT_MPS_EXACT_OBSERVABLE_MAX_QUBITS", 16, 3, 62);
    }

    inline long mps_rdm_mc_samples()
    {
        return env::integer("MIPT_MPS_RDM_MC_SAMPLES", 8192, 0, 1000000000L);
    }

    inline long mps_tmi_dense_max_qubits()
    {
        return env::integer("SIM_TMI_MPS_DENSE_MAX_QUBITS", 16, 4, 30);
    }

    inline void warn_mps_amplitude_path_once(const char *operation, int n, bool monte_carlo, long samples)
    {
        static bool warned = false;
        if (warned)
        {
            return;
        }
        warned = true;
        std::cerr << operation << ": using CUDA-Q backend-native amplitude queries because the compiled target "
                  << MIPT_CUDAQ_BACKEND_NAME << " did not expose a dense rank-1 state tensor.";
        if (monte_carlo)
        {
            std::cerr << " 3-site RDMs are estimated with uniform environment Monte Carlo"
                      << " samples=" << samples << ", n=" << n
                      << "; increase MIPT_MPS_RDM_MC_SAMPLES or set MIPT_MPS_EXACT_OBSERVABLE_MAX_QUBITS>=n for exact summation.";
        }
        else
        {
            std::cerr << " Exact environment summation is used for n=" << n << ".";
        }
        if (amplitude_reverse_bits())
        {
            std::cerr << " MIPT_STATE_AMPLITUDE_REVERSE_BITS=1.";
        }
        std::cerr << '\n';
    }

    inline const char *compiled_cudaq_backend()
    {
        return MIPT_CUDAQ_BACKEND_NAME;
    }

    inline bool compiled_for_tensor_backend()
    {
        return std::strcmp(compiled_cudaq_backend(), "tensornet") == 0 ||
               std::strcmp(compiled_cudaq_backend(), "tensornet-mps") == 0;
    }

    inline bool compiled_for_mps_backend()
    {
        return std::strcmp(compiled_cudaq_backend(), "tensornet-mps") == 0;
    }

    inline bool direct_fermion_boundary_enabled()
    {
        // The direct Jordan-Wigner CZ-string boundary construction removes the
        // long FSWAP conveyor used by the historical closed-boundary fermion
        // implementation.  It is the default now for both exact statevector and
        // experimental tensor/MPS builds because it is an exact gate-count
        // reduction for the same periodic fermionic boundary operation.  Set
        // MIPT_DIRECT_FERMION_BOUNDARY=0 to reproduce the old FSWAP-chain path.
        return env::boolean("MIPT_DIRECT_FERMION_BOUNDARY", true);
    }

    inline void print_backend_banner_once(const char *program_name)
    {
        static bool printed = false;
        if (printed)
        {
            return;
        }
        printed = true;

        std::cerr << program_name << " backend: CUDA-Q target="
                  << compiled_cudaq_backend();
        if (MIPT_CUDAQ_PRECISION == 32 || MIPT_CUDAQ_PRECISION == 64)
        {
            std::cerr << ", statevector_precision=fp" << MIPT_CUDAQ_PRECISION;
        }

        const char *requested = std::getenv("MIPT_TENSOR_BACKEND");
        if (requested != nullptr && *requested != '\0')
        {
            std::cerr << ", MIPT_TENSOR_BACKEND=" << requested;
        }

        if (env::boolean("MIPT_NATIVE_MPS", false) || env::boolean("MIPT_NATIVE_TEBD", false))
        {
            const std::string fallback_abs = env::text("CUDAQ_MPS_ABS_CUTOFF", "1e-10");
            const std::string fallback_rel = env::text("CUDAQ_MPS_RELATIVE_CUTOFF", "1e-10");
            std::cerr << ", native_mps=1"
                      << ", MIPT_NATIVE_MPS_MAX_BOND=" << env::integer("MIPT_NATIVE_MPS_MAX_BOND", env::integer("CUDAQ_MPS_MAX_BOND", 64, 1, 4096), 1, 4096)
                      << ", MIPT_NATIVE_MPS_ABS_CUTOFF=" << env::text("MIPT_NATIVE_MPS_ABS_CUTOFF", fallback_abs.c_str())
                      << ", MIPT_NATIVE_MPS_RELATIVE_CUTOFF=" << env::text("MIPT_NATIVE_MPS_RELATIVE_CUTOFF", fallback_rel.c_str());
        }

        if (compiled_for_mps_backend())
        {
            const char *max_bond = std::getenv("CUDAQ_MPS_MAX_BOND");
            const char *abs_cutoff = std::getenv("CUDAQ_MPS_ABS_CUTOFF");
            const char *rel_cutoff = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF");
            const char *svd_algo = std::getenv("CUDAQ_MPS_SVD_ALGO");
            std::cerr << ", approximate_mps=1";
            if (max_bond != nullptr && *max_bond != '\0')
            {
                std::cerr << ", CUDAQ_MPS_MAX_BOND=" << max_bond;
            }
            if (abs_cutoff != nullptr && *abs_cutoff != '\0')
            {
                std::cerr << ", CUDAQ_MPS_ABS_CUTOFF=" << abs_cutoff;
            }
            if (rel_cutoff != nullptr && *rel_cutoff != '\0')
            {
                std::cerr << ", CUDAQ_MPS_RELATIVE_CUTOFF=" << rel_cutoff;
            }
            if (svd_algo != nullptr && *svd_algo != '\0')
            {
                std::cerr << ", CUDAQ_MPS_SVD_ALGO=" << svd_algo;
            }
            std::cerr << ", MIPT_MPS_EXACT_OBSERVABLE_MAX_QUBITS=" << mps_exact_observable_max_qubits()
                      << ", MIPT_MPS_RDM_MC_SAMPLES=" << mps_rdm_mc_samples()
                      << ", MIPT_MPS_AMPLITUDE_BATCH=" << mps_amplitude_batch_size()
                      << ", MIPT_DIRECT_FERMION_BOUNDARY=" << (direct_fermion_boundary_enabled() ? 1 : 0);
            if (debug_prefix_layers() > 0)
            {
                std::cerr << ", MIPT_DEBUG_PREFIX_LAYERS=" << debug_prefix_layers();
            }
        }
        else if (compiled_for_tensor_backend())
        {
            std::cerr << ", tensor_network=1";
        }
        std::cerr << '\n';
    }

    inline std::string dense_state_required_message(const char *operation)
    {
        std::string msg = operation;
        msg += " requires a dense rank-1 statevector from cudaq::get_state().";
        if (compiled_for_mps_backend())
        {
            msg += " This binary was compiled for the approximate CUDA-Q tensornet-mps target, "
                   "which may not expose a dense state tensor for exact RDM/TMI extraction. "
                   "Use the default nvidia target for exact statevector output, or use the MPS "
                   "target only for exploratory approximate runs after validating tensor-state support.";
        }
        else if (compiled_for_tensor_backend())
        {
            msg += " This binary was compiled for a CUDA-Q tensor-network target. If the target "
                   "does not expose a dense state tensor, the current exact RDM/TMI extractor cannot "
                   "consume it directly.";
        }
        return msg;
    }
} // namespace mipt::backend

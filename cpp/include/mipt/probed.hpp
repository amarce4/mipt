#pragma once

#include "mipt/circuit.hpp"

#include <cudaq.h>

#include <chrono>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mipt::probed
{
struct InitializeKernel
{
    void operator()(int n, int probe_site) __qpu__
    {
        cudaq::qvector q(n + 1);
        h(q[probe_site]);
        cx(q[probe_site], q[n]);
    }
};

struct MmsAdvanceKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    const std::vector<MmsLayer> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(state);
        apply_mms_layers(q, n, layers, closed);
    }
};

struct HaarAdvanceKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    const std::vector<HaarLayer> &layers) __qpu__
    {
        cudaq::qvector q(state);
        apply_haar_layers(q, n, layers);
    }
};

struct RppuAdvanceKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    const std::vector<RppuLayer> &layers) __qpu__
    {
        cudaq::qvector q(state);
        apply_rppu_layers(q, n, layers);
    }
};

struct RfgsAdvanceKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    const std::vector<RfgsLayer> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(state);
        apply_rfgs_layers(q, n, layers, closed);
    }
};

struct RppuParityProbeKernel
{
    void operator()(int n,
                    int probe_site,
                    const std::vector<RppuLayer> &layers) __qpu__
    {
        cudaq::qvector q(n);
        // Exact symmetry encoding of a reference Bell pair: when every system
        // operation preserves computational parity, the omitted reference bit
        // is equal to the system parity.  H on probe_site is the isometric
        // image of (|00>+|11>)/sqrt(2).
        h(q[probe_site]);
        apply_rppu_layers(q, n, layers);
    }
};

struct RfgsParityProbeKernel
{
    void operator()(int n,
                    int probe_site,
                    const std::vector<RfgsLayer> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);
        h(q[probe_site]);
        apply_rfgs_layers(q, n, layers, closed);
    }
};

// Builds one complete random layer history per realization and advances the
// same post-measurement trajectory through selected contiguous layer segments.
// One timestep is exactly one unitary brickwork layer plus its measurement
// layer. No random extra final even layer is added.
class CircuitWorkspace1D
{
  public:
    void reserve(int timesteps)
    {
        if (timesteps < 0)
        {
            throw std::invalid_argument("timesteps must be non-negative.");
        }
        const auto count = static_cast<std::size_t>(timesteps);
        mms_layers_.reserve(count);
        haar_layers_.reserve(count);
        rppu_layers_.reserve(count);
        rfgs_layers_.reserve(count);
        mms_segment_.reserve(count);
        haar_segment_.reserve(count);
        rppu_segment_.reserve(count);
        rfgs_segment_.reserve(count);
    }

    void prepare(int n, int timesteps, double p, CircuitType type)
    {
        if (timesteps < 0)
        {
            throw std::invalid_argument("timesteps must be non-negative.");
        }
        validate_circuit_site_count(type, n, "N");

        prepare_with_probability(
            n,
            timesteps,
            [p](int) { return p; },
            type);
    }

    // Gullans--Huse encoding schedule: scramble without measurements through
    // t0, then evolve with measurement probability p through timesteps.
    void prepare_quench(int n,
                        int timesteps,
                        int t0,
                        double p,
                        CircuitType type)
    {
        if (t0 < 0 || t0 > timesteps)
        {
            throw std::invalid_argument("t0 must lie in [0,timesteps].");
        }
        prepare_with_probability(
            n,
            timesteps,
            [t0, p](int step) { return step < t0 ? 0.0 : p; },
            type);
    }

    template <typename ProbabilityAt>
    void prepare_with_probability(int n,
                                  int timesteps,
                                  ProbabilityAt probability_at,
                                  CircuitType type)
    {
        if (timesteps < 0)
        {
            throw std::invalid_argument("timesteps must be non-negative.");
        }
        validate_circuit_site_count(type, n, "N");

        prepared_n_ = n;
        prepared_timesteps_ = timesteps;
        prepared_type_ = type;

        auto probability = [&](int step)
        {
            const double value = probability_at(step);
            if (value < 0.0 || value > 1.0)
            {
                throw std::invalid_argument(
                    "Layer measurement probability must lie in [0,1].");
            }
            return value;
        };

        switch (type)
        {
        case CircuitType::MMS:
            mms_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_mms_layer(mms_layers_[static_cast<std::size_t>(step)],
                               n, step % 2, probability(step), rng_);
            }
            return;

        case CircuitType::Haar:
            haar_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_haar_layer(haar_layers_[static_cast<std::size_t>(step)],
                                n, step % 2, probability(step), true, rng_);
            }
            return;

        case CircuitType::FermionRPPU:
        case CircuitType::QubitRPPU:
        {
            rppu_layers_.resize(static_cast<std::size_t>(timesteps));
            const bool qubit_boundary = type == CircuitType::QubitRPPU;
            const RppuBoundaryMode boundary_mode = qubit_boundary
                ? RppuBoundaryMode::QubitDirect
                : (backend::direct_fermion_boundary_enabled()
                       ? RppuBoundaryMode::FermionicJWString
                       : RppuBoundaryMode::FermionicFSwap);
            for (int step = 0; step < timesteps; ++step)
            {
                fill_rppu_layer(rppu_layers_[static_cast<std::size_t>(step)],
                                n, (step % 2) != 0, probability(step), true, rng_,
                                boundary_mode);
            }
            return;
        }

        case CircuitType::RFGS:
            rfgs_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_rfgs_layer(rfgs_layers_[static_cast<std::size_t>(step)],
                                n, step % 2, probability(step), rng_, true);
            }
            return;
        }
        throw std::invalid_argument("Unsupported probed circuit type.");
    }

    cudaq::state initialize(int n,
                            int probe_site,
                            std::string_view context = {}) const
    {
        if (probe_site < 0 || probe_site >= n)
        {
            throw std::invalid_argument("probe_site is outside the MIPT circuit.");
        }
        log_start(context, "initialization", n, 0);
        const auto start = std::chrono::steady_clock::now();
        auto state = cudaq::get_state(InitializeKernel{}, n, probe_site);
        log_done(context, "initialization", start);
        return state;
    }

    cudaq::state simulate_parity_encoded_probe(
        int n,
        int probe_site,
        std::string_view context = {}) const
    {
        if (n != prepared_n_ || probe_site < 0 || probe_site >= n)
        {
            throw std::invalid_argument(
                "Parity-encoded probe does not match the prepared circuit.");
        }
        if (!preserves_computational_parity(prepared_type_))
        {
            throw std::invalid_argument(
                "Parity-encoded probes require a parity-preserving circuit.");
        }
        log_start(
            context,
            circuit_type_name(prepared_type_),
            prepared_n_,
            prepared_timesteps_,
            0);
        const auto start = std::chrono::steady_clock::now();
        switch (prepared_type_)
        {
        case CircuitType::FermionRPPU:
        case CircuitType::QubitRPPU:
            return finish_logged(
                cudaq::get_state(
                    RppuParityProbeKernel{},
                    n,
                    probe_site,
                    rppu_layers_),
                context,
                circuit_type_name(prepared_type_),
                start);
        case CircuitType::RFGS:
            return finish_logged(
                cudaq::get_state(
                    RfgsParityProbeKernel{},
                    n,
                    probe_site,
                    rfgs_layers_,
                    true),
                context,
                circuit_type_name(prepared_type_),
                start);
        default:
            break;
        }
        throw std::invalid_argument(
            "Unsupported parity-encoded probe circuit.");
    }

    cudaq::state advance(cudaq::state &state,
                         int first_timestep,
                         int timestep_count,
                         std::string_view context = {})
    {
        validate_segment(first_timestep, timestep_count);
        log_start(context, circuit_type_name(prepared_type_), prepared_n_, timestep_count);
        const auto start = std::chrono::steady_clock::now();

        switch (prepared_type_)
        {
        case CircuitType::MMS:
            return finish_logged(
                run_segment(
                    mms_layers_, mms_segment_, first_timestep, timestep_count,
                    [&](const std::vector<MmsLayer> &layers)
                    {
                        return cudaq::get_state(
                            MmsAdvanceKernel{}, &state, prepared_n_, layers, true);
                    }),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::Haar:
            return finish_logged(
                run_segment(
                    haar_layers_, haar_segment_, first_timestep, timestep_count,
                    [&](const std::vector<HaarLayer> &layers)
                    {
                        return cudaq::get_state(
                            HaarAdvanceKernel{}, &state, prepared_n_, layers);
                    }),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::FermionRPPU:
        case CircuitType::QubitRPPU:
            return finish_logged(
                run_segment(
                    rppu_layers_, rppu_segment_, first_timestep, timestep_count,
                    [&](const std::vector<RppuLayer> &layers)
                    {
                        return cudaq::get_state(
                            RppuAdvanceKernel{}, &state, prepared_n_, layers);
                    }),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::RFGS:
            return finish_logged(
                run_segment(
                    rfgs_layers_, rfgs_segment_, first_timestep, timestep_count,
                    [&](const std::vector<RfgsLayer> &layers)
                    {
                        return cudaq::get_state(
                            RfgsAdvanceKernel{}, &state, prepared_n_, layers, true);
                    }),
                context, circuit_type_name(prepared_type_), start);
        }
        throw std::invalid_argument("Unsupported probed circuit type.");
    }

  private:
    template <typename Layer, typename Runner>
    static cudaq::state run_segment(std::vector<Layer> &source,
                                    std::vector<Layer> &destination,
                                    int first,
                                    int count,
                                    Runner &&runner)
    {
        if (first == 0 && count == static_cast<int>(source.size()))
        {
            // The common fixed-time path consumes the complete prepared
            // history. Pass it directly rather than deep-copying every nested
            // gate and measurement vector into the segment scratch space.
            return runner(source);
        }
        if (count == 1)
        {
            // get_state is synchronous. Swap the selected layer into a reusable
            // one-element payload, run it, then restore the prepared history.
            // This moves nested vectors in O(1) instead of deep-copying every
            // gate and measurement array at every sampled timestep.
            destination.clear();
            destination.resize(1);
            std::swap(destination[0], source[static_cast<std::size_t>(first)]);
            try
            {
                auto result = runner(destination);
                std::swap(destination[0], source[static_cast<std::size_t>(first)]);
                return result;
            }
            catch (...)
            {
                std::swap(destination[0], source[static_cast<std::size_t>(first)]);
                throw;
            }
        }

        // The only normal multi-layer request is the unsampled prefix t_min.
        // It is copied once per realization, not once per sampled timestep.
        const auto begin = source.begin() + first;
        destination.assign(begin, begin + count);
        return runner(destination);
    }

    void validate_segment(int first, int count) const
    {
        if (prepared_n_ <= 0)
        {
            throw std::logic_error("prepare() must be called before advance().");
        }
        if (count <= 0 || first < 0 || first > prepared_timesteps_ - count)
        {
            throw std::invalid_argument("Requested timestep segment is outside the prepared trajectory.");
        }
    }

    static std::string prefix(std::string_view context)
    {
        return context.empty() ? std::string{} : std::string(context) + " ";
    }

    static void log_start(std::string_view context,
                          std::string_view name,
                          int n,
                          int timesteps,
                          int extra_qubits = 1)
    {
        if (backend::verbose_enabled())
        {
            backend::verbose_log(
                prefix(context) + "get_state start: probed " + std::string(name) +
                " system_n=" + std::to_string(n) +
                " total_qubits=" + std::to_string(n + extra_qubits) +
                " timesteps=" + std::to_string(timesteps));
        }
    }

    static void log_done(std::string_view context,
                         std::string_view name,
                         std::chrono::steady_clock::time_point start)
    {
        if (backend::verbose_enabled())
        {
            backend::verbose_log(
                prefix(context) + "get_state done: probed " + std::string(name) +
                " elapsed_s=" + std::to_string(backend::seconds_since(start)));
        }
    }

    static cudaq::state finish_logged(
        cudaq::state state,
        std::string_view context,
        std::string_view name,
        std::chrono::steady_clock::time_point start)
    {
        log_done(context, name, start);
        return state;
    }

    std::mt19937 rng_{std::random_device{}()};
    int prepared_n_ = -1;
    int prepared_timesteps_ = -1;
    CircuitType prepared_type_ = CircuitType::MMS;

    std::vector<MmsLayer> mms_layers_;
    std::vector<HaarLayer> haar_layers_;
    std::vector<RppuLayer> rppu_layers_;
    std::vector<RfgsLayer> rfgs_layers_;

    std::vector<MmsLayer> mms_segment_;
    std::vector<HaarLayer> haar_segment_;
    std::vector<RppuLayer> rppu_segment_;
    std::vector<RfgsLayer> rfgs_segment_;
};

} // namespace mipt::probed

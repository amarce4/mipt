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

        prepared_n_ = n;
        prepared_timesteps_ = timesteps;
        prepared_type_ = type;

        switch (type)
        {
        case CircuitType::MMS:
            mms_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_mms_layer(mms_layers_[static_cast<std::size_t>(step)],
                               n, step % 2, p, rng_);
            }
            return;

        case CircuitType::Haar:
            haar_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_haar_layer(haar_layers_[static_cast<std::size_t>(step)],
                                n, step % 2, p, true, rng_);
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
                                n, (step % 2) != 0, p, true, rng_, boundary_mode);
            }
            return;
        }

        case CircuitType::RFGS:
            rfgs_layers_.resize(static_cast<std::size_t>(timesteps));
            for (int step = 0; step < timesteps; ++step)
            {
                fill_rfgs_layer(rfgs_layers_[static_cast<std::size_t>(step)],
                                n, step % 2, p, rng_, true);
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
            assign_segment(mms_layers_, mms_segment_, first_timestep, timestep_count);
            return finish_logged(
                cudaq::get_state(MmsAdvanceKernel{}, &state, prepared_n_,
                                 mms_segment_, true),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::Haar:
            assign_segment(haar_layers_, haar_segment_, first_timestep, timestep_count);
            return finish_logged(
                cudaq::get_state(HaarAdvanceKernel{}, &state, prepared_n_,
                                 haar_segment_),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::FermionRPPU:
        case CircuitType::QubitRPPU:
            assign_segment(rppu_layers_, rppu_segment_, first_timestep, timestep_count);
            return finish_logged(
                cudaq::get_state(RppuAdvanceKernel{}, &state, prepared_n_,
                                 rppu_segment_),
                context, circuit_type_name(prepared_type_), start);

        case CircuitType::RFGS:
            assign_segment(rfgs_layers_, rfgs_segment_, first_timestep, timestep_count);
            return finish_logged(
                cudaq::get_state(RfgsAdvanceKernel{}, &state, prepared_n_,
                                 rfgs_segment_, true),
                context, circuit_type_name(prepared_type_), start);
        }
        throw std::invalid_argument("Unsupported probed circuit type.");
    }

  private:
    template <typename Layer>
    static void assign_segment(const std::vector<Layer> &source,
                               std::vector<Layer> &destination,
                               int first,
                               int count)
    {
        const auto begin = source.begin() + first;
        destination.assign(begin, begin + count);
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
                          int timesteps)
    {
        if (backend::verbose_enabled())
        {
            backend::verbose_log(
                prefix(context) + "get_state start: probed " + std::string(name) +
                " system_n=" + std::to_string(n) +
                " total_qubits=" + std::to_string(n + 1) +
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

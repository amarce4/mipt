#pragma once

#include "mipt/circuits/mms.hpp"
#include "mipt/circuits/haar.hpp"
#include "mipt/circuits/fermion.hpp"

#ifdef MIPT_ENABLE_CUSV
#include "mipt/cusv/engine.hpp"
#include "mipt/cusv/parity.hpp"
#endif

#include <cudaq.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mipt
{
struct CircuitBuildTiming
{
    std::chrono::steady_clock::time_point frontend_done{};
};

// |0...0> on n qubits. The cuStateVec engine starts from a CUDA-Q-owned buffer
// so that the returned cudaq::state is an ordinary state as far as every
// downstream RDM/entropy consumer is concerned.
struct ZeroStateKernel
{
    void operator()(int n) __qpu__
    {
        cudaq::qvector q(n);
    }
};

class Circuit1D
{
  public:
    virtual ~Circuit1D() = default;
    virtual CircuitType type() const noexcept = 0;
    virtual void reserve(int periods) = 0;
    virtual cudaq::state simulate(int n,
                                  int periods,
                                  double p,
                                  std::string_view context,
                                  CircuitBuildTiming *timing) = 0;

  protected:
    static std::string prefix(std::string_view context)
    {
        return context.empty() ? std::string{} : std::string(context) + " ";
    }

    static void mark_frontend_done(CircuitBuildTiming *timing)
    {
        if (timing != nullptr)
        {
            timing->frontend_done = std::chrono::steady_clock::now();
        }
    }

    static void log_start(std::string_view context, std::string_view name,
                          int n, std::size_t layers)
    {
        if (mipt::backend::verbose_enabled())
        {
            mipt::backend::verbose_log(prefix(context) + "get_state start: " +
                                      std::string(name) + " n=" + std::to_string(n) +
                                      " layers=" + std::to_string(layers));
        }
    }

    static void log_done(std::string_view context, std::string_view name,
                         std::chrono::steady_clock::time_point start)
    {
        if (mipt::backend::verbose_enabled())
        {
            mipt::backend::verbose_log(prefix(context) + "get_state done: " +
                                      std::string(name) + " elapsed_s=" +
                                      std::to_string(mipt::backend::seconds_since(start)));
        }
    }

    static void log_stats(std::string_view context, std::string_view name,
                          const CircuitWorkStats &stats,
                          int n, int periods, double p)
    {
        if (mipt::backend::verbose_enabled())
        {
            const std::string label = prefix(context) + std::string(name);
            mipt::backend::verbose_log(circuit_work_stats_string(
                label.c_str(), stats, n, periods, p));
        }
    }

#ifdef MIPT_ENABLE_CUSV
    // Runs a complete prepared layer history through the cuStateVec engine.
    //
    // The state starts as a CUDA-Q |0...0> and is then mutated in place, so the
    // returned object is a normal cudaq::state whose device buffer CUDA-Q still
    // owns -- every RDM, entropy, and TMI consumer reads it unchanged. Returns
    // nullopt when the engine cannot drive this run (MIPT_CUSV=0, a tensor
    // backend, a CPU state, or a circuit the engine does not implement), and
    // the caller falls back to the CUDA-Q kernel.
    //
    // `build_ops` maps one layer to its operator list; measurement layers are
    // sampled jointly, exactly as in the CUDA-Q path but in two passes instead
    // of two per measured site.
    template <typename Layer, typename BuildOps>
    std::optional<cudaq::state> try_cusv(int n, const std::vector<Layer> &layers,
                                         BuildOps build_ops)
    {
        if (!cusv::enabled())
        {
            return std::nullopt;
        }
        auto state = cudaq::get_state(ZeroStateKernel{}, n);
        const auto tensor = state.get_tensor();
        if (tensor.get_rank() != 1 || tensor.data == nullptr || !state.is_on_gpu() ||
            tensor.get_num_elements() != (std::size_t{1} << n))
        {
            return std::nullopt;
        }
        const bool fp64 = state.get_precision() == cudaq::SimulationState::precision::fp64;
        if (!engine_ || engine_->fp64() != fp64)
        {
            engine_ = std::make_unique<cusv::Engine>(fp64);
        }
        engine_->adopt(tensor.data, n);

        const int block = cusv::max_block_targets();
        for (const auto &layer : layers)
        {
            const auto ops = build_ops(layer);
            engine_->apply_ops(ops.ops, block);
            engine_->measure_layer(ops.measure_sites, measure_rng_);
        }
        return state;
    }

    // Same as try_cusv, but simulates inside the even global-parity sector on
    // n-1 encoded qubits, so every pass moves half the bytes. Only valid for a
    // parity-preserving gate set started from |0...0>; see mipt/cusv/parity.hpp
    // for why the odd half of the state vector is identically zero there.
    //
    // Returns nullopt without having touched a state vector when the encoding
    // does not cover the circuit, so the caller falls back to try_cusv. The
    // whole layer history is rewritten up front for exactly that reason: a
    // refusal half-way through would leave a mutated buffer behind.
    template <typename Layer, typename BuildOps>
    std::optional<cudaq::state> try_cusv_parity(int n, const std::vector<Layer> &layers,
                                                BuildOps build_ops)
    {
        if (!cusv::enabled() || !cusv::parity_encoding_enabled() || n < 4)
        {
            return std::nullopt;
        }

        std::vector<std::vector<cusv::Op>> encoded;
        std::vector<std::vector<int>> measured;
        encoded.reserve(layers.size());
        measured.reserve(layers.size());
        for (const auto &layer : layers)
        {
            auto ops = build_ops(layer);
            auto rewritten = cusv::encode_parity_sector(ops.ops, n);
            if (!rewritten)
            {
                return std::nullopt;
            }
            encoded.push_back(std::move(*rewritten));
            measured.push_back(std::move(ops.measure_sites));
        }

        auto state = cudaq::get_state(ZeroStateKernel{}, n);
        const auto tensor = state.get_tensor();
        if (tensor.get_rank() != 1 || tensor.data == nullptr || !state.is_on_gpu() ||
            tensor.get_num_elements() != (std::size_t{1} << n))
        {
            return std::nullopt;
        }
        const bool fp64 = state.get_precision() == cudaq::SimulationState::precision::fp64;
        if (!engine_ || engine_->fp64() != fp64)
        {
            engine_ = std::make_unique<cusv::Engine>(fp64);
        }
        // CUDA-Q hands back |0...0>, so the low 2^(n-1) amplitudes are already
        // the encoded |0...0> and the upper half is zero -- which is what
        // expand_parity_sector() expects to find there at the end.
        engine_->adopt(tensor.data, n - 1);

        static bool announced = false;
        if (!announced)
        {
            announced = true;
            std::cerr << "parity_sector=1: parity-preserving circuit simulated on 2^" << (n - 1)
                      << " amplitudes; MIPT_CUSV_PARITY=0 restores the full 2^" << n << " path.\n";
        }

        const int block = cusv::max_block_targets();
        for (std::size_t i = 0; i < encoded.size(); ++i)
        {
            engine_->apply_ops(encoded[i], block);
            engine_->measure_layer_encoded(measured[i], n, measure_rng_);
        }
        engine_->expand_parity_sector(n);
        return state;
    }

    std::unique_ptr<cusv::Engine> engine_;
    // Separate stream so measurement outcomes never perturb layer sampling.
    std::mt19937 measure_rng_{std::random_device{}()};
#endif
};

class MmsCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::MMS; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    cudaq::state simulate(int n, int periods, double p, std::string_view context,
                          CircuitBuildTiming *timing) override
    {
        build_mms_layers(layers_, n, periods, p, rng_);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "MMS").c_str());
        log_stats(context, "MMS", circuit_work_stats_mms(layers_, n, true), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "MMS", n, layers_.size());
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        if (auto fast = try_cusv(n, layers_,
                                 [n](const MmsLayer &layer)
                                 { return cusv::build_mms_layer_ops(layer, n, true); }))
        {
            log_done(context, "MMS", start);
            return *fast;
        }
#endif
        auto state = cudaq::get_state(MmsKernel1D{}, n, layers_, true);
        log_done(context, "MMS", start);
        return state;
    }

  private:
    std::mt19937 rng_{std::random_device{}()};
    std::vector<MmsLayer> layers_;
};

class HaarCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::Haar; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods)); }

    cudaq::state simulate(int n, int periods, double p, std::string_view context,
                          CircuitBuildTiming *timing) override
    {
        build_haar_layers(layers_, n, periods, p, rng_, true);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "Haar").c_str());
        mark_frontend_done(timing);
        log_start(context, "Haar", n, layers_.size());
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        if (auto fast = try_cusv(n, layers_,
                                 [n](const HaarLayer &layer)
                                 { return cusv::build_haar_layer_ops(layer, n); }))
        {
            log_done(context, "Haar", start);
            return *fast;
        }
#endif
        auto state = cudaq::get_state(HaarKernel1D{}, n, layers_);
        log_done(context, "Haar", start);
        return state;
    }

  private:
    std::mt19937 rng_{std::random_device{}()};
    std::vector<HaarLayer> layers_;
};

class RppuCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::FermionRPPU; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    cudaq::state simulate(int n, int periods, double p, std::string_view context,
                          CircuitBuildTiming *timing) override
    {
        const bool direct_boundary = mipt::backend::direct_fermion_boundary_enabled();
        build_rppu_layers(layers_, n, periods, p, true, rng_, direct_boundary);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "FermionRPPU").c_str());
        log_stats(context, "FermionRPPU", circuit_work_stats_rppu(layers_), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "FermionRPPU", n, layers_.size());
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        const auto rppu_ops = [n](const RppuLayer &layer)
        { return cusv::build_rppu_layer_ops(layer, n); };
        if (auto fast = try_cusv_parity(n, layers_, rppu_ops))
        {
            log_done(context, "FermionRPPU", start);
            return *fast;
        }
        if (auto fast = try_cusv(n, layers_, rppu_ops))
        {
            log_done(context, "FermionRPPU", start);
            return *fast;
        }
#endif
        auto state = cudaq::get_state(RppuKernel1D{}, n, layers_);
        log_done(context, "FermionRPPU", start);
        return state;
    }

  private:
    std::mt19937 rng_{std::random_device{}()};
    std::vector<RppuLayer> layers_;
};

class RfgsCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::RFGS; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    cudaq::state simulate(int n, int periods, double p, std::string_view context,
                          CircuitBuildTiming *timing) override
    {
        build_rfgs_layers(layers_, n, periods, p, rng_, true);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "RFGS").c_str());
        log_stats(context, "RFGS", circuit_work_stats_rfgs(layers_, n, true), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "RFGS", n, layers_.size());
        const auto start = std::chrono::steady_clock::now();
        auto state = cudaq::get_state(RfgsKernel1D{}, n, layers_, true);
        log_done(context, "RFGS", start);
        return state;
    }

  private:
    std::mt19937 rng_{std::random_device{}()};
    std::vector<RfgsLayer> layers_;
};

class QrppuCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::QubitRPPU; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    cudaq::state simulate(int n, int periods, double p, std::string_view context,
                          CircuitBuildTiming *timing) override
    {
        build_qrppu_layers(layers_, n, periods, p, true, rng_);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "qRPPU").c_str());
        log_stats(context, "qRPPU", circuit_work_stats_rppu(layers_), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "qRPPU", n, layers_.size());
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        const auto rppu_ops = [n](const RppuLayer &layer)
        { return cusv::build_rppu_layer_ops(layer, n); };
        if (auto fast = try_cusv_parity(n, layers_, rppu_ops))
        {
            log_done(context, "qRPPU", start);
            return *fast;
        }
        if (auto fast = try_cusv(n, layers_, rppu_ops))
        {
            log_done(context, "qRPPU", start);
            return *fast;
        }
#endif
        auto state = cudaq::get_state(RppuKernel1D{}, n, layers_);
        log_done(context, "qRPPU", start);
        return state;
    }

  private:
    std::mt19937 rng_{std::random_device{}()};
    std::vector<RppuLayer> layers_;
};

inline std::unique_ptr<Circuit1D> make_circuit_1d(CircuitType type)
{
    switch (type)
    {
    case CircuitType::MMS: return std::make_unique<MmsCircuit1D>();
    case CircuitType::Haar: return std::make_unique<HaarCircuit1D>();
    case CircuitType::FermionRPPU: return std::make_unique<RppuCircuit1D>();
    case CircuitType::RFGS: return std::make_unique<RfgsCircuit1D>();
    case CircuitType::QubitRPPU: return std::make_unique<QrppuCircuit1D>();
    }
    throw std::invalid_argument("Unsupported 1D circuit type.");
}

class CircuitWorkspace1D
{
  public:
    void reserve(int periods)
    {
        reserved_periods_ = periods;
        if (circuit_) circuit_->reserve(periods);
    }

    cudaq::state simulate(int n, int periods, double p, CircuitType type,
                          std::string_view context = {},
                          CircuitBuildTiming *timing = nullptr)
    {
        if (!circuit_ || circuit_->type() != type)
        {
            circuit_ = make_circuit_1d(type);
            circuit_->reserve(reserved_periods_ >= 0 ? reserved_periods_ : periods);
        }
        return circuit_->simulate(n, periods, p, context, timing);
    }

  private:
    int reserved_periods_ = -1;
    std::unique_ptr<Circuit1D> circuit_;
};

} // namespace mipt

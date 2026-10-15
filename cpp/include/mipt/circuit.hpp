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
#include <stdexcept>
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

// One trajectory: the final state, and the logical layer history that produced
// it.
//
// These two must describe the *same* random realization, which is why they
// travel together rather than being recoverable separately. The history is
// derived from the very layer list that is then handed to the simulator, after
// any debug truncation, so there is no way for a caller to pair a state with
// the graph of a different draw. That pairing is the whole basis of the
// spacetime percolation diagnostics in dist_connectivity.hpp: comparing a
// measured entanglement against a graph built from an independent circuit
// would measure nothing at all.
struct TrajectoryResult
{
    cudaq::state state;
    LogicalHistory history;
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

#ifdef MIPT_ENABLE_CUSV
// A read-only window onto a trajectory as the cuStateVec engine runs it.
//
// Exists for one consumer, the stage-2 gate-trace replay (rppu_gate_trace.exe),
// which has to see the state *between* the unitary half and the measurement
// half of chosen layers of a trajectory it regenerates from a recorded seed.
// Regenerating is only worth anything if the outcomes come out the same, and
// they only do if the replay drives the identical engine calls with the
// identically seeded measure_rng_ -- so the replay has to run *this* code, not
// a copy of it. The hook is how it does that without a second simulator.
//
// Zero behaviour change by construction:
//   * with no observer installed (the default, and every production caller)
//     the engine calls are the same calls, in the same order, with the same
//     arguments as before the hook existed;
//   * with one installed, the only differences are the callbacks themselves
//     and an `outcomes` vector handed to measure_layer*(), which those
//     functions only ever write -- they draw exactly the same uniforms from
//     the rng either way (cusv/engine.hpp);
//   * an observer must not mutate the engine. It gets a reference so it can
//     call to_host(); anything else would change the trajectory it is
//     observing.
//
// `layer` indexes the prepared layer list. `encoded` says whether the engine
// currently holds the even-parity-sector state on n-1 qubits (try_cusv_parity)
// or the full 2^n state (try_cusv); `n` is always the full mode count. The
// measured `sites` and their `outcomes` are given in the order the layer lists
// them.
struct LayerObserver
{
    virtual ~LayerObserver() = default;
    virtual void before_unitaries(std::size_t layer, cusv::Engine &engine, bool encoded,
                                  int n) = 0;
    virtual void after_measurements(std::size_t layer, cusv::Engine &engine, bool encoded,
                                    int n, const std::vector<int> &sites,
                                    const std::vector<int> &outcomes) = 0;
};
#endif

// Which engine produced the last trajectory; see Circuit1D::last_engine_path().
enum class EnginePath
{
    None,
    CusvParity,
    Cusv,
    CudaqFallback,
};

class Circuit1D
{
  public:
    virtual ~Circuit1D() = default;
    virtual CircuitType type() const noexcept = 0;
    virtual void reserve(int periods) = 0;
    virtual TrajectoryResult simulate(int n,
                                      int periods,
                                      double p,
                                      std::string_view context,
                                      CircuitBuildTiming *timing) = 0;

    // Pin this trajectory's layer draw.
    //
    // What this does and does not buy: every circuit's *layer history* --
    // which gates are drawn and, decisively, which sites each layer measures
    // -- becomes an exact function of the seed, so the spacetime graph is
    // fully reproducible. Measurement *outcomes* follow only on the cuStateVec
    // path, whose sampling runs off measure_rng_; under the CUDA-Q fallback
    // the runtime draws them from a stream nothing here controls. The graph
    // does not depend on outcomes, so it is reproducible either way.
    void seed(std::uint64_t value)
    {
        std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xffffffffu),
                               static_cast<std::uint32_t>(value >> 32)};
        rng_.seed(sequence);
#ifdef MIPT_ENABLE_CUSV
        const std::uint64_t measurement = seeding::splitmix64(value ^ 0xa0761d0dull);
        std::seed_seq measure_sequence{static_cast<std::uint32_t>(measurement & 0xffffffffu),
                                       static_cast<std::uint32_t>(measurement >> 32)};
        measure_rng_.seed(measure_sequence);
#endif
    }

    // Refuse, rather than take, the CUDA-Q gate-by-gate fallback. Off by
    // default and never set by a production executable. The gate-trace replay
    // sets it because measurement outcomes only follow the seed on the
    // cuStateVec path: a trajectory that silently fell back would be a
    // different trajectory with the right layer draw, which is exactly the
    // failure a replay cannot afford to make quietly. Honoured by the
    // circuits the engine implements (RPPU, qRPPU); see refuse_fallback().
    void require_cusv(bool required) { require_cusv_ = required; }
    EnginePath last_engine_path() const { return last_engine_path_; }

#ifdef MIPT_ENABLE_CUSV
    // Install (or with nullptr, remove) the observer consulted by try_cusv and
    // try_cusv_parity. Not owned. nullptr is the default and the production
    // configuration.
    void set_layer_observer(LayerObserver *observer) { observer_ = observer; }
#endif

  protected:
    bool require_cusv_ = false;
    EnginePath last_engine_path_ = EnginePath::None;

    void refuse_fallback(std::string_view name) const
    {
        if (require_cusv_)
        {
            throw std::runtime_error(
                std::string(name) +
                ": the cuStateVec engine could not run this trajectory and the caller "
                "requires it (the CUDA-Q fallback draws measurement outcomes from a stream "
                "the seed does not control). Check MIPT_CUSV, the CUDA-Q target, and that "
                "the build has CUSV=1.");
        }
    }

    // Shared by every circuit so that seed() has one stream to pin rather than
    // five identical private ones.
    std::mt19937 rng_{std::random_device{}()};

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
        for (std::size_t index = 0; index < layers.size(); ++index)
        {
            const auto ops = build_ops(layers[index]);
            if (observer_ == nullptr)
            {
                engine_->apply_ops(ops.ops, block);
                engine_->measure_layer(ops.measure_sites, measure_rng_);
                continue;
            }
            // Same calls as above; the outcomes vector is write-only inside
            // measure_layer, so the rng draws are identical.
            observer_->before_unitaries(index, *engine_, false, n);
            engine_->apply_ops(ops.ops, block);
            observed_outcomes_.clear();
            engine_->measure_layer(ops.measure_sites, measure_rng_, &observed_outcomes_);
            observer_->after_measurements(index, *engine_, false, n, ops.measure_sites,
                                          observed_outcomes_);
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
            if (observer_ == nullptr)
            {
                engine_->apply_ops(encoded[i], block);
                engine_->measure_layer_encoded(measured[i], n, measure_rng_);
                continue;
            }
            // Same calls as above; see LayerObserver for why the outcomes
            // vector changes nothing.
            observer_->before_unitaries(i, *engine_, true, n);
            engine_->apply_ops(encoded[i], block);
            observed_outcomes_.clear();
            engine_->measure_layer_encoded(measured[i], n, measure_rng_, &observed_outcomes_);
            observer_->after_measurements(i, *engine_, true, n, measured[i], observed_outcomes_);
        }
        engine_->expand_parity_sector(n);
        return state;
    }

    std::unique_ptr<cusv::Engine> engine_;
    // Separate stream so measurement outcomes never perturb layer sampling.
    std::mt19937 measure_rng_{std::random_device{}()};
    LayerObserver *observer_ = nullptr;
    std::vector<int> observed_outcomes_;
#endif
};

class MmsCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::MMS; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    TrajectoryResult simulate(int n, int periods, double p, std::string_view context,
                              CircuitBuildTiming *timing) override
    {
        build_mms_layers(layers_, n, periods, p, rng_);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "MMS").c_str());
        log_stats(context, "MMS", circuit_work_stats_mms(layers_, n, true), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "MMS", n, layers_.size());
        // The graph is derived from the layer list that is about to be
        // simulated -- after any debug truncation -- so state and history are
        // the same realization by construction.
        LogicalHistory history = logical_history_from_mms(layers_, n);
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        if (auto fast = try_cusv(n, layers_,
                                 [n](const MmsLayer &layer)
                                 { return cusv::build_mms_layer_ops(layer, n, true); }))
        {
            log_done(context, "MMS", start);
            return {std::move(*fast), std::move(history)};
        }
#endif
        auto state = cudaq::get_state(MmsKernel1D{}, n, layers_, true);
        log_done(context, "MMS", start);
        return {std::move(state), std::move(history)};
    }

  private:
    std::vector<MmsLayer> layers_;
};

class HaarCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::Haar; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods)); }

    TrajectoryResult simulate(int n, int periods, double p, std::string_view context,
                              CircuitBuildTiming *timing) override
    {
        build_haar_layers(layers_, n, periods, p, rng_, true);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "Haar").c_str());
        mark_frontend_done(timing);
        log_start(context, "Haar", n, layers_.size());
        // The graph is derived from the layer list that is about to be
        // simulated -- after any debug truncation -- so state and history are
        // the same realization by construction.
        LogicalHistory history = logical_history_from_haar(layers_, n);
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        if (auto fast = try_cusv(n, layers_,
                                 [n](const HaarLayer &layer)
                                 { return cusv::build_haar_layer_ops(layer, n); }))
        {
            log_done(context, "Haar", start);
            return {std::move(*fast), std::move(history)};
        }
#endif
        auto state = cudaq::get_state(HaarKernel1D{}, n, layers_);
        log_done(context, "Haar", start);
        return {std::move(state), std::move(history)};
    }

  private:
    std::vector<HaarLayer> layers_;
};

class RppuCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::FermionRPPU; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    TrajectoryResult simulate(int n, int periods, double p, std::string_view context,
                              CircuitBuildTiming *timing) override
    {
        const bool direct_boundary = mipt::backend::direct_fermion_boundary_enabled();
        build_rppu_layers(layers_, n, periods, p, true, rng_, direct_boundary);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "FermionRPPU").c_str());
        log_stats(context, "FermionRPPU", circuit_work_stats_rppu(layers_), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "FermionRPPU", n, layers_.size());
        // The graph is derived from the layer list that is about to be
        // simulated -- after any debug truncation -- so state and history are
        // the same realization by construction.
        LogicalHistory history = logical_history_from_rppu(layers_, n);
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        const auto rppu_ops = [n](const RppuLayer &layer)
        { return cusv::build_rppu_layer_ops(layer, n); };
        if (auto fast = try_cusv_parity(n, layers_, rppu_ops))
        {
            last_engine_path_ = EnginePath::CusvParity;
            log_done(context, "FermionRPPU", start);
            return {std::move(*fast), std::move(history)};
        }
        if (auto fast = try_cusv(n, layers_, rppu_ops))
        {
            last_engine_path_ = EnginePath::Cusv;
            log_done(context, "FermionRPPU", start);
            return {std::move(*fast), std::move(history)};
        }
#endif
        refuse_fallback("FermionRPPU");
        last_engine_path_ = EnginePath::CudaqFallback;
        auto state = cudaq::get_state(RppuKernel1D{}, n, layers_);
        log_done(context, "FermionRPPU", start);
        return {std::move(state), std::move(history)};
    }

    // The layer list the last simulate() drew and ran, after any debug
    // truncation -- the exact list the engine executed, so a replay can rebuild
    // each gate's 4x4 from it (rppu_gate_matrix) rather than redrawing.
    // Already filled in when the first LayerObserver callback fires.
    const std::vector<RppuLayer> &layers() const { return layers_; }

  private:
    std::vector<RppuLayer> layers_;
};

class RfgsCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::RFGS; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    TrajectoryResult simulate(int n, int periods, double p, std::string_view context,
                              CircuitBuildTiming *timing) override
    {
        build_rfgs_layers(layers_, n, periods, p, rng_, true);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "RFGS").c_str());
        log_stats(context, "RFGS", circuit_work_stats_rfgs(layers_, n, true), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "RFGS", n, layers_.size());
        // The graph is derived from the layer list that is about to be
        // simulated -- after any debug truncation -- so state and history are
        // the same realization by construction.
        LogicalHistory history = logical_history_from_rfgs(layers_, n, true);
        const auto start = std::chrono::steady_clock::now();
        auto state = cudaq::get_state(RfgsKernel1D{}, n, layers_, true);
        log_done(context, "RFGS", start);
        return {std::move(state), std::move(history)};
    }

  private:
    std::vector<RfgsLayer> layers_;
};

class QrppuCircuit1D final : public Circuit1D
{
  public:
    CircuitType type() const noexcept override { return CircuitType::QubitRPPU; }
    void reserve(int periods) override { layers_.reserve(static_cast<std::size_t>(2 * periods + 1)); }

    TrajectoryResult simulate(int n, int periods, double p, std::string_view context,
                              CircuitBuildTiming *timing) override
    {
        build_qrppu_layers(layers_, n, periods, p, true, rng_);
        apply_debug_prefix_layer_limit(layers_, (prefix(context) + "qRPPU").c_str());
        log_stats(context, "qRPPU", circuit_work_stats_rppu(layers_), n, periods, p);
        mark_frontend_done(timing);
        log_start(context, "qRPPU", n, layers_.size());
        // The graph is derived from the layer list that is about to be
        // simulated -- after any debug truncation -- so state and history are
        // the same realization by construction.
        LogicalHistory history = logical_history_from_rppu(layers_, n);
        const auto start = std::chrono::steady_clock::now();
#ifdef MIPT_ENABLE_CUSV
        const auto rppu_ops = [n](const RppuLayer &layer)
        { return cusv::build_rppu_layer_ops(layer, n); };
        if (auto fast = try_cusv_parity(n, layers_, rppu_ops))
        {
            last_engine_path_ = EnginePath::CusvParity;
            log_done(context, "qRPPU", start);
            return {std::move(*fast), std::move(history)};
        }
        if (auto fast = try_cusv(n, layers_, rppu_ops))
        {
            last_engine_path_ = EnginePath::Cusv;
            log_done(context, "qRPPU", start);
            return {std::move(*fast), std::move(history)};
        }
#endif
        refuse_fallback("qRPPU");
        last_engine_path_ = EnginePath::CudaqFallback;
        auto state = cudaq::get_state(RppuKernel1D{}, n, layers_);
        log_done(context, "qRPPU", start);
        return {std::move(state), std::move(history)};
    }

  private:
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

    TrajectoryResult simulate(int n, int periods, double p, CircuitType type,
                              std::string_view context = {},
                              CircuitBuildTiming *timing = nullptr)
    {
        ensure(type, periods);
        return circuit_->simulate(n, periods, p, context, timing);
    }

    // Pin the next trajectory's layer draw. Must be called after the circuit
    // exists, so it takes the type it will be used with.
    void seed(CircuitType type, int periods, std::uint64_t value)
    {
        ensure(type, periods);
        circuit_->seed(value);
    }

  private:
    void ensure(CircuitType type, int periods)
    {
        if (!circuit_ || circuit_->type() != type)
        {
            circuit_ = make_circuit_1d(type);
            circuit_->reserve(reserved_periods_ >= 0 ? reserved_periods_ : periods);
        }
    }

    int reserved_periods_ = -1;
    std::unique_ptr<Circuit1D> circuit_;
};

} // namespace mipt

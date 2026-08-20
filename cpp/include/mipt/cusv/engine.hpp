#pragma once

// Direct cuStateVec circuit engine.
//
// This is optimizations #2, #3, and #4:
//
//   #4  The state vector is allocated once and mutated in place. The CUDA-Q
//       path returns a fresh cudaq::state from every get_state() call, and
//       `cudaq::qvector q(state)` re-initialises a new simulator buffer from
//       it, so each advanced timestep paid a full device-to-device copy and
//       kept two live 2^N buffers alive. Here there is one buffer.
//
//   #2  A Jordan-Wigner CZ chain is applied as one diagonal sign pass.
//
//   #3  A whole measurement layer is sampled from its joint distribution with
//       one custatevecAbs2SumArray pass and applied with one
//       custatevecCollapseByBitString pass, instead of ~2 passes per measured
//       site. This is exact: sequential Born sampling with renormalisation
//       samples from the joint distribution (chain rule), and Z-basis
//       projectors on distinct sites commute, so the post-measurement state is
//       the same. measure_layer() returns -log P(joint outcome), which by the
//       chain rule equals the sum of the sequential conditional surprisals
//       free_energy.exe accumulates.

#include "mipt/cuda/cusv_diag.hpp"
#include "mipt/cuda/cusv_parity.hpp"
#include "mipt/cusv/gates.hpp"

#include <custatevec.h>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::cusv
{

inline void check_cusv(custatevecStatus_t status, const char *what)
{
    if (status != CUSTATEVEC_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("cuStateVec ") + what + " failed: " +
                                 custatevecGetErrorString(status));
    }
}

inline void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA ") + what + " failed: " +
                                 cudaGetErrorString(status));
    }
}

// Largest joint measurement block sampled in one pass. A layer with more
// measured sites than this is split into chunks; chunking stays exact because
// the chain rule applies to the chunks as well, and it bounds both the host
// histogram (2^bits doubles) and the smallest representable joint probability.
inline int max_joint_measure_bits()
{
    return static_cast<int>(env::integer("MIPT_CUSV_MEASURE_BITS", 16, 1, 24));
}

// Largest fused gate block (optimization #5). A 2^k x 2^k apply moves the same
// bytes as a 2-target apply but costs 2^(k-1) flop/byte, so the useful block
// size is set by where the GPU crosses from bandwidth-bound to compute-bound:
// k = 2^(k-1) flop/byte vs the machine balance (FLOP/s over B/s).
//
//   GTX 1650    ~4.4 TFLOP/s / 192 GB/s  -> balance ~23 flop/byte -> k <= 5
//   RTX 4080 S  ~50  TFLOP/s / 736 GB/s  -> balance ~68 flop/byte -> k <= 7
//
// Measured on the GTX 1650 (cpp/test/cusv_bench.cpp, n=22, RPPU): k=4 is
// fastest, k=6 and k=8 are slower, matching the estimate. 4 fuses two brickwork
// bonds per pass. Raise MIPT_CUSV_BLOCK_TARGETS to 6 on a card with more
// compute headroom; re-run cusv_bench to confirm before changing it.
inline int max_block_targets()
{
    return static_cast<int>(env::integer("MIPT_CUSV_BLOCK_TARGETS", 4, 2, 12));
}

// Whether the engine should be used at all. MIPT_CUSV=0 restores the CUDA-Q
// gate-by-gate path; the tensor-network targets never expose the dense device
// buffer the engine needs.
inline bool enabled()
{
    static const bool on = env::boolean("MIPT_CUSV", true);
    return on && !backend::compiled_for_tensor_backend();
}

class Engine
{
  public:
    explicit Engine(bool use_fp64) : fp64_(use_fp64)
    {
        check_cusv(custatevecCreate(&handle_), "custatevecCreate");
    }

    ~Engine()
    {
        if (state_ != nullptr) { cudaFree(state_); }
        if (workspace_ != nullptr) { cudaFree(workspace_); }
        if (handle_ != nullptr) { custatevecDestroy(handle_); }
    }

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    bool fp64() const { return fp64_; }
    int qubits() const { return qubits_; }
    void *data() { return active(); }
    const void *data() const { return borrowed_ != nullptr ? borrowed_ : state_; }
    cudaDataType_t data_type() const { return fp64_ ? CUDA_C_64F : CUDA_C_32F; }
    std::size_t dimension() const { return std::size_t{1} << qubits_; }
    std::size_t element_bytes() const { return fp64_ ? 16u : 8u; }

    // Operate on a buffer owned by someone else -- in practice the device
    // pointer inside a live cudaq::state. Nothing is allocated or freed, so the
    // CUDA-Q simulator keeps ownership and every existing consumer that reads
    // `state.get_tensor().data` sees the mutations. This is how optimization #4
    // reaches the apps without a per-timestep state copy.
    void adopt(void *device_ptr, int n_qubits)
    {
        if (device_ptr == nullptr || n_qubits < 1 || n_qubits >= 40)
        {
            throw std::invalid_argument("cusv::Engine::adopt got an invalid buffer.");
        }
        release_owned();
        borrowed_ = device_ptr;
        qubits_ = n_qubits;
    }

    // Allocate (if needed) and set |0...0>.
    void reset(int n_qubits)
    {
        if (n_qubits < 1 || n_qubits >= 40)
        {
            throw std::invalid_argument("cusv::Engine supports 1..39 qubits.");
        }
        borrowed_ = nullptr;
        ensure_capacity(n_qubits);
        qubits_ = n_qubits;
        check_cusv(custatevecInitializeStateVector(handle_, active(), data_type(), qubits_,
                                                   CUSTATEVEC_STATE_VECTOR_TYPE_ZERO),
                   "custatevecInitializeStateVector");
    }

    void set_from_host(const std::vector<std::complex<double>> &host, int n_qubits)
    {
        if (borrowed_ == nullptr)
        {
            ensure_capacity(n_qubits);
        }
        qubits_ = n_qubits;
        if (host.size() != dimension())
        {
            throw std::invalid_argument("Host state size does not match the qubit count.");
        }
        if (fp64_)
        {
            check_cuda(cudaMemcpy(active(), host.data(), dimension() * 16u, cudaMemcpyHostToDevice),
                       "cudaMemcpy(set_from_host fp64)");
        }
        else
        {
            std::vector<std::complex<float>> narrow(host.size());
            std::transform(host.begin(), host.end(), narrow.begin(),
                           [](const std::complex<double> &z)
                           { return std::complex<float>(static_cast<float>(z.real()),
                                                        static_cast<float>(z.imag())); });
            check_cuda(cudaMemcpy(active(), narrow.data(), dimension() * 8u, cudaMemcpyHostToDevice),
                       "cudaMemcpy(set_from_host fp32)");
        }
    }

    void to_host(std::vector<std::complex<double>> &host) const
    {
        host.resize(dimension());
        if (fp64_)
        {
            check_cuda(cudaMemcpy(host.data(), data(), dimension() * 16u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(to_host fp64)");
            return;
        }
        std::vector<std::complex<float>> narrow(dimension());
        check_cuda(cudaMemcpy(narrow.data(), data(), dimension() * 8u, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(to_host fp32)");
        std::transform(narrow.begin(), narrow.end(), host.begin(),
                       [](const std::complex<float> &z)
                       { return std::complex<double>(z.real(), z.imag()); });
    }

    void apply(const Op &op)
    {
        if (op.kind == OpKind::ParityGate)
        {
            const int status = fp64_ ? mipt_cuda_parity_gate_f64(active(), qubits_, op.targets[0],
                                                                 op.parity_even.data(),
                                                                 op.parity_odd.data())
                                     : mipt_cuda_parity_gate_f32(active(), qubits_, op.targets[0],
                                                                 op.parity_even.data(),
                                                                 op.parity_odd.data());
            if (status != 0)
            {
                throw std::runtime_error(std::string("Parity-gate pass failed: ") +
                                         mipt_cuda_cusv_parity_last_error());
            }
            return;
        }
        if (op.kind == OpKind::JwString)
        {
            const int status = fp64_
                                   ? mipt_cuda_jw_string_f64(active(), qubits_, op.string_control,
                                                             op.string_mask)
                                   : mipt_cuda_jw_string_f32(active(), qubits_, op.string_control,
                                                             op.string_mask);
            if (status != 0)
            {
                throw std::runtime_error(std::string("JW-string pass failed: ") +
                                         mipt_cuda_cusv_diag_last_error());
            }
            return;
        }
        apply_matrix(op.targets, op.matrix);
    }

    void apply_matrix(const std::vector<int> &targets, const DenseMatrix &matrix)
    {
        if (static_cast<int>(targets.size()) != matrix.targets)
        {
            throw std::invalid_argument("Target count does not match the matrix rank.");
        }
        const std::vector<std::int32_t> t(targets.begin(), targets.end());
        const cudaDataType_t matrix_type = fp64_ ? CUDA_C_64F : CUDA_C_32F;
        const custatevecComputeType_t compute =
            fp64_ ? CUSTATEVEC_COMPUTE_64F : CUSTATEVEC_COMPUTE_32F;

        const void *matrix_data = nullptr;
        std::vector<std::complex<float>> narrow;
        if (fp64_)
        {
            matrix_data = matrix.values.data();
        }
        else
        {
            narrow.resize(matrix.values.size());
            std::transform(matrix.values.begin(), matrix.values.end(), narrow.begin(),
                           [](const Complex &z)
                           { return std::complex<float>(static_cast<float>(z.real()),
                                                        static_cast<float>(z.imag())); });
            matrix_data = narrow.data();
        }

        std::size_t needed = 0;
        check_cusv(custatevecApplyMatrixGetWorkspaceSize(
                       handle_, data_type(), qubits_, matrix_data, matrix_type,
                       CUSTATEVEC_MATRIX_LAYOUT_ROW, /*adjoint=*/0,
                       static_cast<std::uint32_t>(t.size()), /*nControls=*/0, compute, &needed),
                   "custatevecApplyMatrixGetWorkspaceSize");
        ensure_workspace(needed);

        check_cusv(custatevecApplyMatrix(handle_, active(), data_type(), qubits_, matrix_data,
                                         matrix_type, CUSTATEVEC_MATRIX_LAYOUT_ROW, /*adjoint=*/0,
                                         t.data(), static_cast<std::uint32_t>(t.size()),
                                         /*controls=*/nullptr, /*controlBitValues=*/nullptr, 0,
                                         compute, workspace_, workspace_bytes_),
                   "custatevecApplyMatrix");
    }

    // Applies every op in order. Blocking (#5) is applied here so callers do
    // not have to remember to do it.
    void apply_ops(const std::vector<Op> &ops, int block_targets)
    {
        const auto blocked = block_disjoint_ops(ops, block_targets);
        for (const auto &op : blocked)
        {
            apply(op);
        }
    }

    // Samples the joint outcome of `sites` and collapses onto it.
    // Returns -log P(joint outcome) -- the record surprisal for this layer.
    // `outcomes` receives the sampled bit for each site, in the order given.
    template <typename Rng>
    double measure_layer(const std::vector<int> &sites, Rng &rng,
                         std::vector<int> *outcomes = nullptr)
    {
        if (outcomes != nullptr) { outcomes->assign(sites.size(), 0); }
        if (sites.empty()) { return 0.0; }

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const int chunk_bits = max_joint_measure_bits();
        double surprisal = 0.0;

        for (std::size_t begin = 0; begin < sites.size(); begin += static_cast<std::size_t>(chunk_bits))
        {
            const std::size_t end =
                std::min(sites.size(), begin + static_cast<std::size_t>(chunk_bits));
            const std::vector<std::int32_t> ordering(sites.begin() + static_cast<std::ptrdiff_t>(begin),
                                                     sites.begin() + static_cast<std::ptrdiff_t>(end));
            const std::size_t bins = std::size_t{1} << ordering.size();

            abs2_.assign(bins, 0.0);
            check_cusv(custatevecAbs2SumArray(handle_, active(), data_type(), qubits_, abs2_.data(),
                                              ordering.data(),
                                              static_cast<std::uint32_t>(ordering.size()), nullptr,
                                              nullptr, 0),
                       "custatevecAbs2SumArray");

            double total = 0.0;
            for (const double v : abs2_) { total += v; }
            if (!(total > 0.0) || !std::isfinite(total))
            {
                throw std::runtime_error("Measurement layer found a zero-norm state vector.");
            }

            // Sample the joint outcome from the exact joint distribution.
            const double draw = uniform(rng) * total;
            double running = 0.0;
            std::size_t chosen = bins - 1;
            for (std::size_t i = 0; i < bins; ++i)
            {
                running += abs2_[i];
                if (draw < running)
                {
                    chosen = i;
                    break;
                }
            }
            double branch = abs2_[chosen];
            if (!(branch > 0.0))
            {
                // Numerically starved bin: fall back to the most probable one.
                chosen = static_cast<std::size_t>(
                    std::max_element(abs2_.begin(), abs2_.end()) - abs2_.begin());
                branch = abs2_[chosen];
                if (!(branch > 0.0))
                {
                    throw std::runtime_error("Measurement layer has no outcome with positive weight.");
                }
            }

            std::vector<std::int32_t> bit_string(ordering.size());
            for (std::size_t i = 0; i < ordering.size(); ++i)
            {
                bit_string[i] = static_cast<std::int32_t>((chosen >> i) & std::size_t{1});
                if (outcomes != nullptr) { (*outcomes)[begin + i] = bit_string[i]; }
            }

            surprisal += -std::log(branch / total);

            check_cusv(custatevecCollapseByBitString(handle_, active(), data_type(), qubits_,
                                                     bit_string.data(), ordering.data(),
                                                     static_cast<std::uint32_t>(ordering.size()),
                                                     branch),
                       "custatevecCollapseByBitString");
        }
        return surprisal;
    }

    // ------------------------------------------------- parity-sector encoding

    // Measures the derived site n-1, which in the encoded picture is
    // parity(e). Returns its contribution to the layer surprisal.
    template <typename Rng>
    double measure_derived_parity(Rng &rng, int *outcome)
    {
        double weight[2] = {0.0, 0.0};
        const int status = fp64_ ? mipt_cuda_parity_abs2_f64(data(), qubits_, weight)
                                 : mipt_cuda_parity_abs2_f32(data(), qubits_, weight);
        if (status != 0)
        {
            throw std::runtime_error(std::string("Parity measurement failed: ") +
                                     mipt_cuda_cusv_parity_last_error());
        }
        const double total = weight[0] + weight[1];
        if (!(total > 0.0) || !std::isfinite(total))
        {
            throw std::runtime_error("Parity measurement found a zero-norm state vector.");
        }

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        int bit = (uniform(rng) * total < weight[0]) ? 0 : 1;
        if (!(weight[bit] > 0.0))
        {
            // Numerically starved branch: fall back to the heavier one.
            bit = (weight[0] >= weight[1]) ? 0 : 1;
        }
        if (!(weight[bit] > 0.0))
        {
            throw std::runtime_error("Parity measurement has no outcome with positive weight.");
        }

        const int collapse = fp64_
                                 ? mipt_cuda_parity_collapse_f64(active(), qubits_, bit, weight[bit])
                                 : mipt_cuda_parity_collapse_f32(active(), qubits_, bit, weight[bit]);
        if (collapse != 0)
        {
            throw std::runtime_error(std::string("Parity collapse failed: ") +
                                     mipt_cuda_cusv_parity_last_error());
        }
        if (outcome != nullptr) { *outcome = bit; }
        return -std::log(weight[bit] / total);
    }

    // A whole measurement layer in the encoded picture. Sites below the
    // derived one go through the ordinary joint path; the derived one follows
    // as its own chunk, which is exact for the same chain-rule reason
    // measure_layer() already splits wide layers into chunks.
    template <typename Rng>
    double measure_layer_encoded(const std::vector<int> &sites, int full_qubits, Rng &rng,
                                 std::vector<int> *outcomes = nullptr)
    {
        if (outcomes != nullptr) { outcomes->assign(sites.size(), 0); }
        if (sites.empty()) { return 0.0; }

        const int derived = full_qubits - 1;
        std::vector<int> plain;
        plain.reserve(sites.size());
        std::size_t derived_slot = sites.size();
        for (std::size_t i = 0; i < sites.size(); ++i)
        {
            if (sites[i] == derived) { derived_slot = i; }
            else { plain.push_back(sites[i]); }
        }

        double surprisal = 0.0;
        if (!plain.empty())
        {
            std::vector<int> plain_outcomes;
            surprisal +=
                measure_layer(plain, rng, outcomes != nullptr ? &plain_outcomes : nullptr);
            if (outcomes != nullptr)
            {
                std::size_t k = 0;
                for (std::size_t i = 0; i < sites.size(); ++i)
                {
                    if (i != derived_slot) { (*outcomes)[i] = plain_outcomes[k++]; }
                }
            }
        }
        if (derived_slot < sites.size())
        {
            int bit = 0;
            surprisal += measure_derived_parity(rng, &bit);
            if (outcomes != nullptr) { (*outcomes)[derived_slot] = bit; }
        }
        return surprisal;
    }

    // Expands the encoded state sitting in the low half of the adopted 2^n
    // buffer into the full state vector, in place, and re-adopts it at full
    // width. Every downstream consumer then sees an ordinary state.
    void expand_parity_sector(int full_qubits)
    {
        if (qubits_ != full_qubits - 1)
        {
            throw std::invalid_argument("expand_parity_sector width does not match the state.");
        }
        const int status = fp64_ ? mipt_cuda_parity_expand_f64(active(), full_qubits)
                                 : mipt_cuda_parity_expand_f32(active(), full_qubits);
        if (status != 0)
        {
            throw std::runtime_error(std::string("Parity expansion failed: ") +
                                     mipt_cuda_cusv_parity_last_error());
        }
        qubits_ = full_qubits;
    }

    // Squared norm of the whole state; used by the self-checks.
    double squared_norm()
    {
        double total = 0.0;
        check_cusv(custatevecAbs2SumArray(handle_, active(), data_type(), qubits_, &total, nullptr, 0,
                                          nullptr, nullptr, 0),
                   "custatevecAbs2SumArray(norm)");
        return total;
    }

    void synchronize() const { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

  private:
    void *active()
    {
        void *p = borrowed_ != nullptr ? borrowed_ : state_;
        if (p == nullptr)
        {
            throw std::runtime_error("cusv::Engine has no state vector; call reset() or adopt().");
        }
        return p;
    }

    void release_owned()
    {
        if (state_ != nullptr)
        {
            cudaFree(state_);
            state_ = nullptr;
            capacity_bytes_ = 0;
        }
    }

    void ensure_capacity(int n_qubits)
    {
        const std::size_t bytes = (std::size_t{1} << n_qubits) * element_bytes();
        if (bytes <= capacity_bytes_ && state_ != nullptr)
        {
            return;
        }
        if (state_ != nullptr)
        {
            check_cuda(cudaFree(state_), "cudaFree(state)");
            state_ = nullptr;
            capacity_bytes_ = 0;
        }
        check_cuda(cudaMalloc(&state_, bytes), "cudaMalloc(state)");
        capacity_bytes_ = bytes;
    }

    void ensure_workspace(std::size_t bytes)
    {
        if (bytes <= workspace_bytes_)
        {
            return;
        }
        if (workspace_ != nullptr)
        {
            check_cuda(cudaFree(workspace_), "cudaFree(workspace)");
            workspace_ = nullptr;
            workspace_bytes_ = 0;
        }
        check_cuda(cudaMalloc(&workspace_, bytes), "cudaMalloc(workspace)");
        workspace_bytes_ = bytes;
    }

    custatevecHandle_t handle_ = nullptr;
    void *state_ = nullptr;      // owned; null while borrowing
    void *borrowed_ = nullptr;   // not owned, never freed
    std::size_t capacity_bytes_ = 0;
    void *workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
    int qubits_ = 0;
    bool fp64_ = false;
    std::vector<double> abs2_;
};

} // namespace mipt::cusv

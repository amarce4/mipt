#pragma once

#include "mipt/tmi/linalg.hpp"
#include "mipt/circuit.hpp"
#include "mipt/rdm.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/tmi_cuda_svd.hpp"
#include "mipt/cuda/tmi_cuda_rdm.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::tmi
{
inline std::vector<int> lower_block_positions(std::uint32_t block_qubits)
{
    std::vector<int> positions;
    positions.reserve(block_qubits);
    for (std::uint32_t i = 0; i < block_qubits; ++i)
    {
        positions.push_back(static_cast<int>(i));
    }
    return positions;
}

inline std::vector<int> upper_block_positions(std::uint32_t block_qubits)
{
    std::vector<int> positions;
    positions.reserve(block_qubits);
    for (std::uint32_t i = 0; i < block_qubits; ++i)
    {
        positions.push_back(static_cast<int>(block_qubits + i));
    }
    return positions;
}

struct TmiPlans
{
    TracePlan ab_to_a;
    TracePlan ab_to_b;
    TracePlan ac_to_c;
};

inline TmiPlans make_tmi_plans(std::uint32_t block_qubits, TraceMode mode)
{
    const int pair_modes = static_cast<int>(2u * block_qubits);
    return {
        make_trace_plan(pair_modes, lower_block_positions(block_qubits), mode),
        make_trace_plan(pair_modes, upper_block_positions(block_qubits), mode),
        make_trace_plan(pair_modes, upper_block_positions(block_qubits), mode),
    };
}

inline std::vector<int> cyclic_mode_range(int begin, int count, int n)
{
    if (n <= 0)
    {
        throw std::invalid_argument("cyclic_mode_range requires positive n.");
    }
    if (count < 0 || count > n)
    {
        throw std::invalid_argument("Invalid cyclic subsystem block size.");
    }

    std::vector<int> modes;
    modes.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        int q = (begin + i) % n;
        if (q < 0)
        {
            q += n;
        }
        modes.push_back(q);
    }
    return modes;
}

inline std::vector<TmiMatrixTerm> tmi_terms_1d_cycle(int n, int cycle_offset)
{
    if (n < 4 || (n % 4) != 0)
    {
        throw std::invalid_argument("TMI output requires N=L >= 4 and divisible by 4.");
    }

    const int block = n / 4;
    if (cycle_offset < 0 || cycle_offset >= block)
    {
        throw std::invalid_argument("TMI cycle offset is out of range.");
    }

    const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
    const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
    const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
    const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);

    return {
        {"AB", concatenate_modes(A, B)},
        {"AC", concatenate_modes(A, C)},
        {"BC", concatenate_modes(B, C)},
        {"D", D},
    };
}

inline int tmi_cycle_count(int n, bool all_cycles)
{
    return all_cycles ? (n / 4) : 1;
}

struct TmiWorkspace
{
    EntropyWorkspace entropy;
    std::vector<C64> rho_a;
    std::vector<C64> rho_b;
    std::vector<C64> rho_c;
    std::vector<C64> direct_matrix;

    StateEntropyWorkspace entropy_a;
    StateEntropyWorkspace entropy_b;
    StateEntropyWorkspace entropy_c;
    StateEntropyWorkspace entropy_d;
    // Reused for AB, AC, and BC half-cut Schmidt SVDs.  For N=24 this
    // avoids retaining three separate 4096x4096 complex work matrices.
    StateEntropyWorkspace entropy_half;
    bool cuda_svd_disabled = false;
    int cuda_svd_failure_status = 0;
    bool cuda_small_rdm_disabled = false;
    int cuda_small_rdm_failure_status = 0;
    std::vector<double> device_rdm_ri;
    std::vector<std::complex<double>> host_state_f64;
    std::vector<std::complex<float>> host_state_f32;
};

inline double tmi_from_payload(const std::vector<double> &payload,
                        std::uint32_t block_qubits,
                        const std::vector<TmiMatrixTerm> &terms,
                        const TmiPlans &plans,
                        TmiWorkspace &workspace)
{
    if (terms.size() != 4)
    {
        throw std::invalid_argument("Online TMI expects four terms: AB, AC, BC, D.");
    }

    std::array<const double *, 4> term_ptrs{};
    std::size_t offset = 0;
    for (std::size_t i = 0; i < terms.size(); ++i)
    {
        term_ptrs[i] = payload.data() + offset;
        offset += term_value_count(terms[i]);
    }
    if (offset != payload.size())
    {
        throw std::runtime_error("Internal TMI payload-size mismatch.");
    }

    const std::size_t block_dim = std::size_t{1} << block_qubits;
    const std::size_t pair_dim = std::size_t{1} << (2u * block_qubits);
    const EntropyOrder order = tmi_entropy_order();

    partial_trace_ri_ptr_to(term_ptrs[0], plans.ab_to_a, workspace.rho_a);
    partial_trace_ri_ptr_to(term_ptrs[0], plans.ab_to_b, workspace.rho_b);
    partial_trace_ri_ptr_to(term_ptrs[1], plans.ac_to_c, workspace.rho_c);

    const double s_a = entropy_hermitian_inplace(workspace.rho_a, block_dim, workspace.entropy, order);
    const double s_b = entropy_hermitian_inplace(workspace.rho_b, block_dim, workspace.entropy, order);
    const double s_c = entropy_hermitian_inplace(workspace.rho_c, block_dim, workspace.entropy, order);
    const double s_d = entropy_ri_ptr(term_ptrs[3], block_dim, workspace.direct_matrix, workspace.entropy, order);
    const double s_ab = entropy_ri_ptr(term_ptrs[0], pair_dim, workspace.direct_matrix, workspace.entropy, order);
    const double s_ac = entropy_ri_ptr(term_ptrs[1], pair_dim, workspace.direct_matrix, workspace.entropy, order);
    const double s_bc = entropy_ri_ptr(term_ptrs[2], pair_dim, workspace.direct_matrix, workspace.entropy, order);

    return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
}

template <typename Real>
inline double tmi_from_statevector(const std::complex<Real> *psi,
                            int n,
                            std::uint32_t block_qubits,
                            int cycle_offset,
                            bool fermion_trace,
                            TmiWorkspace &workspace)
{
    const int block = static_cast<int>(block_qubits);
    if (cycle_offset < 0 || cycle_offset >= block)
    {
        throw std::invalid_argument("TMI cycle offset is out of range.");
    }

    const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
    const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
    const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
    const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
    const std::vector<int> AB = concatenate_modes(A, B);
    const std::vector<int> AC = concatenate_modes(A, C);
    const std::vector<int> BC = concatenate_modes(B, C);

    const EntropyOrder order = tmi_entropy_order();

    const double s_a = entropy_subsystem_rdm_from_state(
        psi, n, A, fermion_trace, workspace.entropy_a, order);
    const double s_b = entropy_subsystem_rdm_from_state(
        psi, n, B, fermion_trace, workspace.entropy_b, order);
    const double s_c = entropy_subsystem_rdm_from_state(
        psi, n, C, fermion_trace, workspace.entropy_c, order);
    const double s_d = entropy_subsystem_rdm_from_state(
        psi, n, D, fermion_trace, workspace.entropy_d, order);

    const double s_ab = entropy_subsystem_svd_from_state(
        psi, n, AB, fermion_trace, workspace.entropy_half, order);
    const double s_ac = entropy_subsystem_svd_from_state(
        psi, n, AC, fermion_trace, workspace.entropy_half, order);
    const double s_bc = entropy_subsystem_svd_from_state(
        psi, n, BC, fermion_trace, workspace.entropy_half, order);

    return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
}


inline bool entropy_subsystem_svd_from_device_state(const void *device_state_vector,
                                             bool device_state_is_fp64,
                                             int n,
                                             const std::vector<int> &kept_modes,
                                             bool fermion_trace,
                                             double &entropy_out,
                                             int &status_out)
{
    status_out = 0;
#ifdef MIPT_ENABLE_CUDA_RHO
    if (device_state_vector == nullptr || kept_modes.empty() ||
        kept_modes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        status_out = -1;
        return false;
    }

    // The Renyi-2 entry point shares the Schmidt gather but stops at the Gram
    // matrix, so it never allocates -- or needs -- the eigensolver workspace.
    const bool renyi2 = tmi_entropy_order() == EntropyOrder::Renyi2;
    const int mode_count = static_cast<int>(kept_modes.size());
    const int trace_flag = fermion_trace ? 1 : 0;
    const int status =
        renyi2 ? (device_state_is_fp64
                      ? mipt_cuda_renyi2_f64(device_state_vector, n, kept_modes.data(),
                                             mode_count, trace_flag, &entropy_out)
                      : mipt_cuda_renyi2_f32(device_state_vector, n, kept_modes.data(),
                                             mode_count, trace_flag, &entropy_out))
               : (device_state_is_fp64
                      ? mipt_cuda_entropy_svd_f64(device_state_vector, n, kept_modes.data(),
                                                  mode_count, trace_flag, &entropy_out)
                      : mipt_cuda_entropy_svd_f32(device_state_vector, n, kept_modes.data(),
                                                  mode_count, trace_flag, &entropy_out));
    status_out = status;
    return status == 0 && std::isfinite(entropy_out);
#else
    (void)device_state_vector;
    (void)device_state_is_fp64;
    (void)n;
    (void)kept_modes;
    (void)fermion_trace;
    (void)entropy_out;
    status_out = -1;
    return false;
#endif
}

inline bool device_small_rdm_enabled()
{
    // On by default since 2026-07-29.  This was off because the old device
    // kernel launched one block per output element and re-streamed the whole
    // environment for each, costing 2*2^(N+kept) loads -- 128x redundant for a
    // kept=6 quarter at N=24 -- which made a full GPU->host statevector copy
    // plus host reduction the cheaper option.  tmi_cuda_rdm.cu now gathers the
    // state once and reduces with cuBLAS herk, so the device path wins at every
    // size and the per-sample state copy disappears.
    //
    // Set SIM_TMI_DEVICE_SMALL_RDM=0 to fall back to the host path.
    static const bool enabled = mipt::env::boolean("SIM_TMI_DEVICE_SMALL_RDM", true);
    return enabled;
}

inline bool entropy_subsystem_rdm_from_device_state(const void *device_state_vector,
                                             bool device_state_is_fp64,
                                             int n,
                                             const std::vector<int> &kept_modes,
                                             bool fermion_trace,
                                             double &entropy_out,
                                             int &status_out,
                                             TmiWorkspace &workspace)
{
    status_out = 0;
#ifdef MIPT_ENABLE_CUDA_RHO
    if (device_state_vector == nullptr || kept_modes.empty() ||
        kept_modes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        status_out = -1;
        return false;
    }

    const std::size_t dim = std::size_t{1} << kept_modes.size();
    if (dim == 0 || dim > (std::numeric_limits<std::size_t>::max() / dim / 2u))
    {
        status_out = -1;
        return false;
    }
    workspace.device_rdm_ri.assign(2u * dim * dim, 0.0);

    const int status = device_state_is_fp64
                           ? mipt_cuda_rdm_subsystem_f64(device_state_vector,
                                                         n,
                                                         kept_modes.data(),
                                                         static_cast<int>(kept_modes.size()),
                                                         fermion_trace ? 1 : 0,
                                                         workspace.device_rdm_ri.data())
                           : mipt_cuda_rdm_subsystem_f32(device_state_vector,
                                                         n,
                                                         kept_modes.data(),
                                                         static_cast<int>(kept_modes.size()),
                                                         fermion_trace ? 1 : 0,
                                                         workspace.device_rdm_ri.data());
    status_out = status;
    if (status != 0)
    {
        return false;
    }
    entropy_out = entropy_ri_ptr(workspace.device_rdm_ri.data(),
                                 dim,
                                 workspace.direct_matrix,
                                 workspace.entropy,
                                 tmi_entropy_order());
    return std::isfinite(entropy_out);
#else
    (void)device_state_vector;
    (void)device_state_is_fp64;
    (void)n;
    (void)kept_modes;
    (void)fermion_trace;
    (void)entropy_out;
    (void)workspace;
    status_out = -1;
    return false;
#endif
}

inline bool tmi_from_device_state_hybrid_gpu(const void *device_state_vector,
                                      bool device_state_is_fp64,
                                      int n,
                                      std::uint32_t block_qubits,
                                      int cycle_offset,
                                      bool fermion_trace,
                                      TmiWorkspace &workspace,
                                      double &tmi_out)
{
    if (!device_small_rdm_enabled() || workspace.cuda_small_rdm_disabled ||
        device_state_vector == nullptr)
    {
        return false;
    }

    const int block = static_cast<int>(block_qubits);
    if (cycle_offset < 0 || cycle_offset >= block)
    {
        throw std::invalid_argument("TMI cycle offset is out of range.");
    }

    const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
    const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
    const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
    const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
    const std::vector<int> AB = concatenate_modes(A, B);
    const std::vector<int> AC = concatenate_modes(A, C);
    const std::vector<int> BC = concatenate_modes(B, C);

    auto small_entropy = [&](const std::vector<int> &modes, double &entropy) -> bool {
        int status = 0;
        if (entropy_subsystem_rdm_from_device_state(device_state_vector,
                                                    device_state_is_fp64,
                                                    n,
                                                    modes,
                                                    fermion_trace,
                                                    entropy,
                                                    status,
                                                    workspace))
        {
            return true;
        }
        workspace.cuda_small_rdm_disabled = true;
        workspace.cuda_small_rdm_failure_status = status;
        return false;
    };

    auto half_entropy = [&](const std::vector<int> &modes, double &entropy) -> bool {
        int status = 0;
        if (!workspace.cuda_svd_disabled &&
            should_try_cuda_svd_for_problem(n, modes.size()) &&
            entropy_subsystem_svd_from_device_state(device_state_vector,
                                                    device_state_is_fp64,
                                                    n,
                                                    modes,
                                                    fermion_trace,
                                                    entropy,
                                                    status))
        {
            return true;
        }
        if (!workspace.cuda_svd_disabled && status != 0)
        {
            workspace.cuda_svd_disabled = true;
            workspace.cuda_svd_failure_status = status;
        }
        return false;
    };

    double s_a = 0.0;
    double s_b = 0.0;
    double s_c = 0.0;
    double s_d = 0.0;
    double s_ab = 0.0;
    double s_ac = 0.0;
    double s_bc = 0.0;
    if (!small_entropy(A, s_a) || !small_entropy(B, s_b) ||
        !small_entropy(C, s_c) || !small_entropy(D, s_d) ||
        !half_entropy(AB, s_ab) || !half_entropy(AC, s_ac) ||
        !half_entropy(BC, s_bc))
    {
        return false;
    }

    tmi_out = s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    return true;
}

inline bool tmi_values_from_device_state_hybrid_gpu(const void *device_state_vector,
                                             bool device_state_is_fp64,
                                             int n,
                                             std::uint32_t block_qubits,
                                             int cycle_count,
                                             bool fermion_trace,
                                             TmiWorkspace &workspace,
                                             std::vector<double> &values)
{
    values.clear();
    values.reserve(static_cast<std::size_t>(cycle_count));
    for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
    {
        double tmi = 0.0;
        if (!tmi_from_device_state_hybrid_gpu(device_state_vector,
                                              device_state_is_fp64,
                                              n,
                                              block_qubits,
                                              cycle_offset,
                                              fermion_trace,
                                              workspace,
                                              tmi))
        {
            values.clear();
            return false;
        }
        values.push_back(tmi);
    }
    return true;
}

template <typename Real>
inline double tmi_from_statevector_hybrid_gpu_half_svd(const std::complex<Real> *host_psi,
                                                const void *device_state_vector,
                                                bool device_state_is_fp64,
                                                int n,
                                                std::uint32_t block_qubits,
                                                int cycle_offset,
                                                bool fermion_trace,
                                                TmiWorkspace &workspace)
{
    const int block = static_cast<int>(block_qubits);
    if (cycle_offset < 0 || cycle_offset >= block)
    {
        throw std::invalid_argument("TMI cycle offset is out of range.");
    }

    const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
    const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
    const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
    const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
    const std::vector<int> AB = concatenate_modes(A, B);
    const std::vector<int> AC = concatenate_modes(A, C);
    const std::vector<int> BC = concatenate_modes(B, C);

    const EntropyOrder order = tmi_entropy_order();

    const double s_a = entropy_subsystem_rdm_from_state(
        host_psi, n, A, fermion_trace, workspace.entropy_a, order);
    const double s_b = entropy_subsystem_rdm_from_state(
        host_psi, n, B, fermion_trace, workspace.entropy_b, order);
    const double s_c = entropy_subsystem_rdm_from_state(
        host_psi, n, C, fermion_trace, workspace.entropy_c, order);
    const double s_d = entropy_subsystem_rdm_from_state(
        host_psi, n, D, fermion_trace, workspace.entropy_d, order);

    auto half_entropy = [&](const std::vector<int> &modes) -> double {
        double entropy = 0.0;
        int status = 0;
        if (!workspace.cuda_svd_disabled &&
            should_try_cuda_svd_for_problem(n, modes.size()) &&
            entropy_subsystem_svd_from_device_state(device_state_vector,
                                                    device_state_is_fp64,
                                                    n,
                                                    modes,
                                                    fermion_trace,
                                                    entropy,
                                                    status))
        {
            return entropy;
        }

        if (!workspace.cuda_svd_disabled && status != 0)
        {
            workspace.cuda_svd_disabled = true;
            workspace.cuda_svd_failure_status = status;
        }
        return entropy_subsystem_svd_from_state(
            host_psi, n, modes, fermion_trace, workspace.entropy_half, order);
    };

    const double s_ab = half_entropy(AB);
    const double s_ac = half_entropy(AC);
    const double s_bc = half_entropy(BC);

    return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
}

template <typename Real>
inline std::vector<double> tmi_values_from_statevector_hybrid_gpu_half_svd(const std::complex<Real> *host_psi,
                                                                    const void *device_state_vector,
                                                                    bool device_state_is_fp64,
                                                                    int n,
                                                                    std::uint32_t block_qubits,
                                                                    int cycle_count,
                                                                    bool fermion_trace,
                                                                    TmiWorkspace &workspace)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(cycle_count));
    for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
    {
        values.push_back(tmi_from_statevector_hybrid_gpu_half_svd(host_psi,
                                                                  device_state_vector,
                                                                  device_state_is_fp64,
                                                                  n,
                                                                  block_qubits,
                                                                  cycle_offset,
                                                                  fermion_trace,
                                                                  workspace));
    }
    return values;
}

template <typename Real>
inline std::vector<double> tmi_values_from_statevector_fast(const std::complex<Real> *psi,
                                                     int n,
                                                     std::uint32_t block_qubits,
                                                     int cycle_count,
                                                     bool fermion_trace,
                                                     TmiWorkspace &workspace)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(cycle_count));
    for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
    {
        values.push_back(tmi_from_statevector(psi,
                                              n,
                                              block_qubits,
                                              cycle_offset,
                                              fermion_trace,
                                              workspace));
    }
    return values;
}

template <typename Real>
inline std::vector<double> tmi_values_from_statevector_payload(const std::complex<Real> *psi,
                                                        int n,
                                                        std::uint32_t block_qubits,
                                                        int cycle_count,
                                                        bool fermion_trace,
                                                        const TmiPlans &plans,
                                                        TmiWorkspace &workspace)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(cycle_count));
    std::vector<double> payload;

    for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
    {
        const std::vector<TmiMatrixTerm> terms = tmi_terms_1d_cycle(n, cycle_offset);
        tmi_density_matrices_from_host_statevector(psi, n, terms, payload, fermion_trace);
        values.push_back(tmi_from_payload(payload, block_qubits, terms, plans, workspace));
    }
    return values;
}


inline void dense_state_from_cudaq_amplitudes(cudaq::state &state,
                                       int n,
                                       std::vector<std::complex<double>> &out)
{
    if (n > static_cast<int>(mipt::backend::mps_tmi_dense_max_qubits()))
    {
        throw std::runtime_error(
            "CUDA-Q tensor/MPS target did not expose a dense state tensor for TMI. "
            "The fallback can reconstruct a dense state from backend-native amplitudes only up to "
            "SIM_TMI_MPS_DENSE_MAX_QUBITS. Increase that variable for small validation jobs, "
            "or use the default nvidia statevector target for exact high-N TMI.");
    }
    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dim = static_cast<std::size_t>(dim_u64);
    out.resize(dim);
    double norm2 = 0.0;
    const std::uint64_t batch_size = static_cast<std::uint64_t>(mipt::backend::mps_amplitude_batch_size());
    std::vector<std::uint64_t> basis_indices;
    std::vector<std::complex<double>> amplitudes;
    if (mipt::backend::verbose_enabled())
    {
        mipt::backend::verbose_log(
            "TMI dense reconstruction: n=" + std::to_string(n) +
            " dim=" + std::to_string(dim_u64) +
            " amplitude_batch=" + std::to_string(batch_size));
    }
    for (std::uint64_t idx0 = 0; idx0 < dim_u64; idx0 += batch_size)
    {
        const std::uint64_t chunk = std::min<std::uint64_t>(batch_size, dim_u64 - idx0);
        basis_indices.clear();
        basis_indices.reserve(static_cast<std::size_t>(chunk));
        for (std::uint64_t k = 0; k < chunk; ++k)
        {
            basis_indices.push_back(idx0 + k);
        }
        cudaq_state_amplitudes_by_indices(state, n, basis_indices, amplitudes);
        for (std::uint64_t k = 0; k < chunk; ++k)
        {
            const auto amp = amplitudes[static_cast<std::size_t>(k)];
            out[static_cast<std::size_t>(idx0 + k)] = amp;
            norm2 += std::norm(amp);
        }
    }
    if (norm2 > 0.0 && std::isfinite(norm2))
    {
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (auto &amp : out)
        {
            amp *= inv_norm;
        }
    }
    mipt::backend::warn_mps_amplitude_path_once("TMI dense-state reconstruction", n, false, 0);
}

inline std::vector<double> tmi_values_from_cudaq_state_amplitude_dense(cudaq::state &state,
                                                                int n,
                                                                std::uint32_t block_qubits,
                                                                int cycle_count,
                                                                bool fermion_trace,
                                                                TmiWorkspace &workspace)
{
    dense_state_from_cudaq_amplitudes(state, n, workspace.host_state_f64);
    return tmi_values_from_statevector_fast(workspace.host_state_f64.data(),
                                            n,
                                            block_qubits,
                                            cycle_count,
                                            fermion_trace,
                                            workspace);
}

inline std::vector<double> tmi_values_from_cudaq_state_payload_amplitude_dense(cudaq::state &state,
                                                                        int n,
                                                                        std::uint32_t block_qubits,
                                                                        int cycle_count,
                                                                        bool fermion_trace,
                                                                        const TmiPlans &plans,
                                                                        TmiWorkspace &workspace)
{
    dense_state_from_cudaq_amplitudes(state, n, workspace.host_state_f64);
    return tmi_values_from_statevector_payload(workspace.host_state_f64.data(),
                                               n,
                                               block_qubits,
                                               cycle_count,
                                               fermion_trace,
                                               plans,
                                               workspace);
}

inline std::vector<double> tmi_values_from_cudaq_state_fast(cudaq::state &state,
                                                     int n,
                                                     std::uint32_t block_qubits,
                                                     int cycle_count,
                                                     bool fermion_trace,
                                                     TmiWorkspace &workspace)
{
    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dim = static_cast<std::size_t>(dim_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();

    if (mipt::backend::verbose_enabled())
    {
        mipt::backend::verbose_log(
            "tmi_values_from_cudaq_state_fast: n=" + std::to_string(n) +
            " num_tensors=" + std::to_string(state.get_num_tensors()) +
            " tensor0_rank=" + std::to_string(tensor.get_rank()) +
            " tensor0_elements=" + std::to_string(tensor.get_num_elements()) +
            " tensor0_data=" + std::to_string(tensor.data != nullptr ? 1 : 0) +
            " state_on_gpu=" + std::to_string(state.is_on_gpu() ? 1 : 0));
    }

    const bool has_dense_rank1_tensor =
        tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
    if (!has_dense_rank1_tensor)
    {
        return tmi_values_from_cudaq_state_amplitude_dense(
            state, n, block_qubits, cycle_count, fermion_trace, workspace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        if (state.is_on_gpu())
        {
            if (tensor.data != nullptr)
            {
                std::vector<double> device_values;
                if (tmi_values_from_device_state_hybrid_gpu(tensor.data,
                                                            true,
                                                            n,
                                                            block_qubits,
                                                            cycle_count,
                                                            fermion_trace,
                                                            workspace,
                                                            device_values))
                {
                    return device_values;
                }
            }
            workspace.host_state_f64.resize(dim);
            state.to_host(workspace.host_state_f64.data(), workspace.host_state_f64.size());
            if (tensor.data != nullptr)
            {
                return tmi_values_from_statevector_hybrid_gpu_half_svd(workspace.host_state_f64.data(),
                                                                       tensor.data,
                                                                       true,
                                                                       n,
                                                                       block_qubits,
                                                                       cycle_count,
                                                                       fermion_trace,
                                                                       workspace);
            }
            return tmi_values_from_statevector_fast(workspace.host_state_f64.data(),
                                                    n,
                                                    block_qubits,
                                                    cycle_count,
                                                    fermion_trace,
                                                    workspace);
        }
        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }
        return tmi_values_from_statevector_fast(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n,
            block_qubits,
            cycle_count,
            fermion_trace,
            workspace);
    }

    if (state.is_on_gpu())
    {
        if (tensor.data != nullptr)
        {
            std::vector<double> device_values;
            if (tmi_values_from_device_state_hybrid_gpu(tensor.data,
                                                        false,
                                                        n,
                                                        block_qubits,
                                                        cycle_count,
                                                        fermion_trace,
                                                        workspace,
                                                        device_values))
            {
                return device_values;
            }
        }
        workspace.host_state_f32.resize(dim);
        state.to_host(workspace.host_state_f32.data(), workspace.host_state_f32.size());
        if (tensor.data != nullptr)
        {
            return tmi_values_from_statevector_hybrid_gpu_half_svd(workspace.host_state_f32.data(),
                                                                   tensor.data,
                                                                   false,
                                                                   n,
                                                                   block_qubits,
                                                                   cycle_count,
                                                                   fermion_trace,
                                                                   workspace);
        }
        return tmi_values_from_statevector_fast(workspace.host_state_f32.data(),
                                                n,
                                                block_qubits,
                                                cycle_count,
                                                fermion_trace,
                                                workspace);
    }
    if (tensor.data == nullptr)
    {
        throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
    }
    return tmi_values_from_statevector_fast(
        reinterpret_cast<const std::complex<float> *>(tensor.data),
        n,
        block_qubits,
        cycle_count,
        fermion_trace,
        workspace);
}

inline std::vector<double> tmi_values_from_cudaq_state_payload(cudaq::state &state,
                                                        int n,
                                                        std::uint32_t block_qubits,
                                                        int cycle_count,
                                                        bool fermion_trace,
                                                        const TmiPlans &plans,
                                                        TmiWorkspace &workspace)
{
    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dim = static_cast<std::size_t>(dim_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();

    if (mipt::backend::verbose_enabled())
    {
        mipt::backend::verbose_log(
            "tmi_values_from_cudaq_state_fast: n=" + std::to_string(n) +
            " num_tensors=" + std::to_string(state.get_num_tensors()) +
            " tensor0_rank=" + std::to_string(tensor.get_rank()) +
            " tensor0_elements=" + std::to_string(tensor.get_num_elements()) +
            " tensor0_data=" + std::to_string(tensor.data != nullptr ? 1 : 0) +
            " state_on_gpu=" + std::to_string(state.is_on_gpu() ? 1 : 0));
    }

    const bool has_dense_rank1_tensor =
        tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
    if (!has_dense_rank1_tensor)
    {
        return tmi_values_from_cudaq_state_payload_amplitude_dense(
            state, n, block_qubits, cycle_count, fermion_trace, plans, workspace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        if (state.is_on_gpu())
        {
            workspace.host_state_f64.resize(dim);
            state.to_host(workspace.host_state_f64.data(), workspace.host_state_f64.size());
            return tmi_values_from_statevector_payload(workspace.host_state_f64.data(),
                                                       n,
                                                       block_qubits,
                                                       cycle_count,
                                                       fermion_trace,
                                                       plans,
                                                       workspace);
        }
        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }
        return tmi_values_from_statevector_payload(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n,
            block_qubits,
            cycle_count,
            fermion_trace,
            plans,
            workspace);
    }

    if (state.is_on_gpu())
    {
        workspace.host_state_f32.resize(dim);
        state.to_host(workspace.host_state_f32.data(), workspace.host_state_f32.size());
        return tmi_values_from_statevector_payload(workspace.host_state_f32.data(),
                                                   n,
                                                   block_qubits,
                                                   cycle_count,
                                                   fermion_trace,
                                                   plans,
                                                   workspace);
    }
    if (tensor.data == nullptr)
    {
        throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
    }
    return tmi_values_from_statevector_payload(
        reinterpret_cast<const std::complex<float> *>(tensor.data),
        n,
        block_qubits,
        cycle_count,
        fermion_trace,
        plans,
        workspace);
}

inline bool cudaq_state_has_device_tensor(cudaq::state &state)
{
#ifdef MIPT_ENABLE_CUDA_RHO
    if (!state.is_on_gpu())
    {
        return false;
    }
    const auto tensor = state.get_tensor();
    return tensor.data != nullptr && tensor.get_rank() == 1;
#else
    (void)state;
    return false;
#endif
}

inline bool host_svd_available_for_state(cudaq::state &state)
{
    const auto precision = state.get_precision();
    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return lapack_runtime::zgesvd() != nullptr;
    }
    return lapack_runtime::cgesvd() != nullptr || lapack_runtime::zgesvd() != nullptr;
}

inline bool fast_state_tmi_available(cudaq::state &state, int n, std::uint32_t block_qubits)
{
    // The fast state-vector path always needs zheevd_ for the small A/B/C/D
    // RDM entropies.  Half-cut entropies can use either host SVD or, when the
    // state is still on GPU and CUDA_RHO is enabled, the cuSOLVER device SVD.
    // The N=8/N=12 small-N CPU override must be included here; otherwise a
    // GPU state with no host SVD could incorrectly enter the fast path and
    // fail after the CUDA half-SVD is intentionally skipped.
    if (lapack_runtime::zheevd() == nullptr)
    {
        return false;
    }
    if (host_svd_available_for_state(state))
    {
        return true;
    }
    const std::size_t half_cut_kept_modes = 2u * static_cast<std::size_t>(block_qubits);
    return cudaq_state_has_device_tensor(state) &&
           should_try_cuda_svd_for_problem(n, half_cut_kept_modes) &&
           device_small_rdm_enabled();
}

} // namespace mipt::tmi

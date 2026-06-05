#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include "density_io.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <complex>
#include <cmath>
#include <set>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>

#ifdef MIPT_ENABLE_CUDA_RHO
#include "rho_reduce.hpp"
#endif

namespace
{
    constexpr int RHO3_QUBITS = 3;
    constexpr int RHO3_DIM = 1 << RHO3_QUBITS; // 8
    constexpr std::size_t RHO3_VALUES = 2u * RHO3_DIM * RHO3_DIM;
    using Subsystem = std::array<int, RHO3_QUBITS>;

    inline std::uint64_t checked_pow2_u64(int exponent)
    {
        if (exponent < 0 || exponent >= 63)
        {
            throw std::invalid_argument("Invalid qubit count: 2^n would overflow uint64_t.");
        }
        return std::uint64_t{1} << exponent;
    }

    std::vector<Subsystem> periodic_ring_three_site_subsystems(int n)
    {
        if (n < RHO3_QUBITS)
        {
            throw std::invalid_argument("Three-qubit RDM output requires n >= 3.");
        }

        std::vector<Subsystem> subsystems;
        subsystems.reserve(static_cast<std::size_t>(n));
        for (int start = 0; start < n; ++start)
        {
            subsystems.push_back({start, (start + 1) % n, (start + 2) % n});
        }
        return subsystems;
    }

    std::vector<Subsystem> three_site_subsystems_2d(int x, int y)
    {
        std::vector<Subsystem> subsystems;
        subsystems.reserve(static_cast<std::size_t>(x*y));
        for (int i = 0; i < y; ++i) // horizontal subsystems
        {
            for (int j = 0; j < x; ++j)
            {
                subsystems.push_back({i*x + j, i*x + (j + 1)%x, i*x + (j + 2)%x});
            }
        }
        for (int i = 0; i < x; ++i) // vertical subsystems
        {
            for (int j = 0; j < y; ++j)
            {
                subsystems.push_back({j*x + i, ((j + 1)%y)*x + i, ((j + 2)%y)*x + i});
            }
        }
        return subsystems;
    }

    std::vector<Subsystem> terminal_three_site_subsystem(int n)
    {
        if (n < RHO3_QUBITS)
        {
            throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
        }
        return {{n - 3, n - 2, n - 1}};
    }

    std::vector<int> flatten_subsystems(const std::vector<Subsystem> &subsystems)
    {
        std::vector<int> flat;
        flat.reserve(subsystems.size() * RHO3_QUBITS);
        for (const auto &subsystem : subsystems)
        {
            flat.insert(flat.end(), subsystem.begin(), subsystem.end());
        }
        return flat;
    }

    template <typename Real>
    void reduced_density_complex_from_host_statevector(
        const std::complex<Real> *psi,
        int n,
        const std::vector<Subsystem> &subsystems,
        double *rho_ri)
    {
        if (psi == nullptr || rho_ri == nullptr)
        {
            throw std::invalid_argument("Null state-vector or density-matrix pointer.");
        }
        if (n < RHO3_QUBITS || subsystems.empty())
        {
            throw std::invalid_argument("Invalid retained subsystem definition.");
        }

        const std::size_t output_values = subsystems.size() * RHO3_VALUES;
        std::fill(rho_ri, rho_ri + output_values, 0.0);
        const std::uint64_t env_dim = checked_pow2_u64(n - RHO3_QUBITS);

        // CUDA-Q qubit b maps to bit b in the linear statevector index.
        // Reduced-basis bit k maps to subsystem[k]. This preserves the
        // requested cyclic ordering for wraparound triples such as {n-1,0,1}.
        for (std::size_t s = 0; s < subsystems.size(); ++s)
        {
            const auto &subsystem = subsystems[s];
            for (int i = 0; i < RHO3_QUBITS; ++i)
            {
                if (subsystem[i] < 0 || subsystem[i] >= n)
                {
                    throw std::invalid_argument("Subsystem qubit index is out of range.");
                }
                for (int j = 0; j < i; ++j)
                {
                    if (subsystem[i] == subsystem[j])
                    {
                        throw std::invalid_argument("Subsystem contains duplicate qubits.");
                    }
                }
            }

            std::array<std::uint64_t, RHO3_DIM> retained_offsets{};
            for (int local_basis = 0; local_basis < RHO3_DIM; ++local_basis)
            {
                for (int k = 0; k < RHO3_QUBITS; ++k)
                {
                    retained_offsets[local_basis] |=
                        (static_cast<std::uint64_t>((local_basis >> k) & 1)
                         << subsystem[k]);
                }
            }

            for (std::uint64_t env = 0; env < env_dim; ++env)
            {
                std::uint64_t base = 0;
                int env_bit = 0;
                for (int q = 0; q < n; ++q)
                {
                    if (q != subsystem[0] && q != subsystem[1] && q != subsystem[2])
                    {
                        base |= ((env >> env_bit) & std::uint64_t{1}) << q;
                        ++env_bit;
                    }
                }

                for (int row = 0; row < RHO3_DIM; ++row)
                {
                    const auto a = psi[base | retained_offsets[row]];
                    for (int col = 0; col < RHO3_DIM; ++col)
                    {
                        const auto b = psi[base | retained_offsets[col]];
                        const auto value = a * std::conj(b);
                        const std::size_t out =
                            s * RHO3_VALUES +
                            2u * static_cast<std::size_t>(row * RHO3_DIM + col);
                        rho_ri[out + 0] += static_cast<double>(std::real(value));
                        rho_ri[out + 1] += static_cast<double>(std::imag(value));
                    }
                }
            }
        }
    }

    void cudaq_three_site_density_matrices(
        cudaq::state &state,
        int n,
        const std::vector<Subsystem> &subsystems,
        double *rho_ri)
    {
        if (rho_ri == nullptr || subsystems.empty())
        {
            throw std::invalid_argument("Density-matrix output definition is empty.");
        }
        if (n < RHO3_QUBITS)
        {
            throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
        }

        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
        {
            throw std::runtime_error("Expected a contiguous rank-1 CUDA-Q statevector tensor.");
        }

#ifdef MIPT_ENABLE_CUDA_RHO
        if (state.is_on_gpu())
        {
            const std::vector<int> flat = flatten_subsystems(subsystems);
            int status = 0;
            if (precision == cudaq::SimulationState::precision::fp64)
            {
                status = mipt_cuda_rho3_subsystems_complex_f64(
                    tensor.data, n, flat.data(),
                    static_cast<int>(subsystems.size()), rho_ri);
            }
            else
            {
                status = mipt_cuda_rho3_subsystems_complex_f32(
                    tensor.data, n, flat.data(),
                    static_cast<int>(subsystems.size()), rho_ri);
            }
            if (status != 0)
            {
                throw std::runtime_error(
                    "CUDA three-qubit reduced-density-matrix kernel failed; status=" +
                    std::to_string(status));
            }
            return;
        }
#endif

        if (state.is_on_gpu())
        {
            // Without the custom reduction kernel, copy the statevector once
            // and form every requested retained-subsystem matrix on the host.
            if (precision == cudaq::SimulationState::precision::fp64)
            {
                std::vector<std::complex<double>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                reduced_density_complex_from_host_statevector(
                    host_state.data(), n, subsystems, rho_ri);
            }
            else
            {
                std::vector<std::complex<float>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                reduced_density_complex_from_host_statevector(
                    host_state.data(), n, subsystems, rho_ri);
            }
            return;
        }

        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            reduced_density_complex_from_host_statevector(
                reinterpret_cast<const std::complex<double> *>(tensor.data),
                n, subsystems, rho_ri);
        }
        else
        {
            reduced_density_complex_from_host_statevector(
                reinterpret_cast<const std::complex<float> *>(tensor.data),
                n, subsystems, rho_ri);
        }
    }

    mipt_io::DensityFileMetadata density_metadata(
        int dimension,
        int total_qubits,
        int periods,
        int realizations,
        int res,
        double p_min,
        double p_max,
        int grid_x,
        int grid_y,
        const std::vector<Subsystem> &subsystems)
    {
        mipt_io::DensityFileMetadata metadata;
        metadata.spatial_dimension = static_cast<std::uint32_t>(dimension);
        metadata.total_qubits = static_cast<std::uint32_t>(total_qubits);
        metadata.kept_qubits = RHO3_QUBITS;
        metadata.matrix_dimension = RHO3_DIM;
        metadata.periods = static_cast<std::uint32_t>(periods);
        metadata.realizations = static_cast<std::uint32_t>(realizations);
        metadata.resolution = static_cast<std::uint32_t>(res);
        metadata.grid_x = static_cast<std::uint32_t>(grid_x);
        metadata.grid_y = static_cast<std::uint32_t>(grid_y);
        metadata.subsystem_count = static_cast<std::uint32_t>(subsystems.size());
        metadata.record_count =
            static_cast<std::uint64_t>(realizations) *
            static_cast<std::uint64_t>(res);
        metadata.p_min = p_min;
        metadata.p_max = p_max;
        metadata.subsystem_qubits.reserve(subsystems.size() * RHO3_QUBITS);
        for (const auto &subsystem : subsystems)
        {
            for (int q : subsystem)
            {
                metadata.subsystem_qubits.push_back(static_cast<std::uint32_t>(q));
            }
        }
        return metadata;
    }
}


template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}

struct LayerData
{
    int start = 0;

    std::vector<int> measure_flags;

    // Exactly one of these should be 1 for each qubit.
    std::vector<int> rot_x_flags;
    std::vector<int> rot_y_flags;
    std::vector<int> rot_xy_flags;
};

struct MIPTKernel_1D
{
    void operator()(int n,
                    const std::vector<LayerData> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            for (int i = start; i < n - 1; i += 2)
            {
                // Local rotation on q[i]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[i + 1]
                if (layers[layer].rot_x_flags[i + 1])
                {
                    rx(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_y_flags[i + 1])
                {
                    ry(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_xy_flags[i + 1])
                {
                    rz(-0.78539816339744830962, q[i + 1]);
                    rx(1.57079632679489661923, q[i + 1]);
                    rz(0.78539816339744830962, q[i + 1]);
                }

                // exp(-i*pi/4 X_i X_{i+1})
                h(q[i]);
                h(q[i + 1]);

                cx(q[i], q[i + 1]);
                rz(1.57079632679489661923, q[i + 1]);
                cx(q[i], q[i + 1]);

                h(q[i]);
                h(q[i + 1]);
            }

            // closed odd-layer closure: bond (n - 1, 0)
            if (closed && start == 1 && n > 2)
            {
                int i = n - 1;
                int j = 0;

                // Local rotation on q[n - 1]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[0]
                if (layers[layer].rot_x_flags[j])
                {
                    rx(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_y_flags[j])
                {
                    ry(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_xy_flags[j])
                {
                    rz(-0.78539816339744830962, q[j]);
                    rx(1.57079632679489661923, q[j]);
                    rz(0.78539816339744830962, q[j]);
                }

                // exp(-i*pi/4 X_{n-1} X_0)
                h(q[i]);
                h(q[j]);

                cx(q[i], q[j]);
                rz(1.57079632679489661923, q[j]);
                cx(q[i], q[j]);

                h(q[i]);
                h(q[j]);
            }

            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

struct MIPTKernel_2D
{
    void operator()(int x, int y,
                    const std::vector<LayerData> &layers,
                    bool closed) __qpu__
    {
        int n = x * y;
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            if (layer % 4 == 0 || layer % 4 == 1)
            {
                for (int i = 0; i < y; ++i)
                {
                    for (int j = start; j < x-1; j += 2)
                    {
                        int idx = i * x + j;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx + 1]
                        if (layers[layer].rot_x_flags[idx + 1])
                        {
                            rx(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_y_flags[idx + 1])
                        {
                            ry(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_xy_flags[idx + 1])
                        {
                            rz(-0.78539816339744830962, q[idx + 1]);
                            rx(1.57079632679489661923, q[idx + 1]);
                            rz(0.78539816339744830962, q[idx + 1]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx + 1]);

                        cx(q[idx], q[idx + 1]);
                        rz(1.57079632679489661923, q[idx + 1]);
                        cx(q[idx], q[idx + 1]);

                        h(q[idx]);
                        h(q[idx + 1]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && start == 1 && x > 2)
                    {
                        int k = i*x;
                        int l = i*x + x - 1;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < x; ++i)
                {
                    int x_start = (i % 2 == 0) ? 0 : 1;
                    x_start = (x_start + layers[layer].start) % 2;
                    for (int j = x_start; j < y-1; j += 2)
                    {
                        int idx = j * y + i;
                        int idx_next = (j + 1) * y + i;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx_next]
                        if (layers[layer].rot_x_flags[idx_next])
                        {
                            rx(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_y_flags[idx_next])
                        {
                            ry(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_xy_flags[idx_next])
                        {
                            rz(-0.78539816339744830962, q[idx_next]);
                            rx(1.57079632679489661923, q[idx_next]);
                            rz(0.78539816339744830962, q[idx_next]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx_next]);

                        cx(q[idx], q[idx_next]);
                        rz(1.57079632679489661923, q[idx_next]);
                        cx(q[idx], q[idx_next]);

                        h(q[idx]);
                        h(q[idx_next]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && x_start == 1 && y > 2)
                    {
                        int k = i;
                        int l = (y-1)*y + i;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
                    }
                }
            }
            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

LayerData make_mipt_layer(int n,
                          int start,
                          double p,
                          std::mt19937 &rng)
{
    std::uniform_int_distribution<int> axis_dist(0, 2);
    std::bernoulli_distribution measure_dist(p);

    LayerData layer;
    layer.start = start;

    layer.measure_flags.resize(n);

    layer.rot_x_flags.resize(n);
    layer.rot_y_flags.resize(n);
    layer.rot_xy_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        int axis = axis_dist(rng);

        layer.rot_x_flags[q] = 0;
        layer.rot_y_flags[q] = 0;
        layer.rot_xy_flags[q] = 0;

        if (axis == 0)
        {
            layer.rot_x_flags[q] = 1;
        }
        else if (axis == 1)
        {
            layer.rot_y_flags[q] = 1;
        }
        else
        {
            layer.rot_xy_flags[q] = 1;
        }

        layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
    }

    return layer;
}

std::vector<LayerData> mipt_frontend(int n,
                                     int periods,
                                     double p)
{
    std::mt19937 rng(std::random_device{}());

    std::vector<LayerData> layers;
    layers.reserve(2 * periods + 1);

    for (int period = 0; period < periods; ++period)
    {
        layers.push_back(make_mipt_layer(n, 0, p, rng)); // even layer
        layers.push_back(make_mipt_layer(n, 1, p, rng)); // odd layer
    }

    // MIPT-style final even layer with 50% measurement probability.
    layers.push_back(make_mipt_layer(n, 0, 0.5, rng));

    return layers;
}



void run_1d(int n, int periods, int realizations, int res, double p_min, double p_max)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires n >= 3.");
    }

    const std::vector<Subsystem> subsystems = periodic_ring_three_site_subsystems(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info.csv");

    // Keep the existing info.csv format unchanged.
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "\n";

    mipt_io::DensityMatrixWriter rho3_file(
        "rho3.bin",
        density_metadata(1, n, periods, realizations, res,
                         p_min, p_max, n, 1, subsystems));

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; ++r)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            auto layers = mipt_frontend(n, periods, p);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            auto state = cudaq::get_state(MIPTKernel_1D{}, n, layers, true);

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);
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

                const std::chrono::duration<double> t_layer = lap1 - start;
                const std::chrono::duration<double> t_qc = lap2 - lap1;
                const std::chrono::duration<double> t_dm = lap3 - lap2;
                const std::chrono::duration<double> t_save = lap4 - lap3;
                std::cout << "Layer, QC, DM, Save times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_dm.count() << "s, "
                          << t_save.count() << "s\n";
            }

            const auto circ_time_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> circ_time_diff =
                circ_time_end - circ_time_start;
            if (r != 0)
            {
                sec_per_circ = ((r - 1) * sec_per_circ +
                                circ_time_diff.count()) / r;
            }
            else
            {
                sec_per_circ = circ_time_diff.count();
            }

            const double time_estimate = (realizations - r) * sec_per_circ;
            std::cout << "\rAverage circuits/second: "
                      << std::round((1 / sec_per_circ) * 100.0) / 100.0
                      << ", Realisations ETA: "
                      << static_cast<int>(time_estimate)
                      << "s               " << std::flush;
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved " << subsystems.size()
              << " cyclic three-qubit reduced density matrices per trajectory to rho3.bin.\n";
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

    mipt_io::DensityMatrixWriter rho3_file(
        "rho3_2d.bin",
        density_metadata(2, n, periods, realizations, res,
                         p_min, p_max, x, y, subsystems));

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; ++r)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            auto layers = mipt_frontend(n, periods, p);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            auto state = cudaq::get_state(MIPTKernel_2D{}, x, y, layers, true);

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);
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

                const std::chrono::duration<double> t_layer = lap1 - start;
                const std::chrono::duration<double> t_qc = lap2 - lap1;
                const std::chrono::duration<double> t_dm = lap3 - lap2;
                const std::chrono::duration<double> t_save = lap4 - lap3;
                std::cout << "Layer, QC, DM, Save times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_dm.count() << "s, "
                          << t_save.count() << "s\n";
            }

            const auto circ_time_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> circ_time_diff =
                circ_time_end - circ_time_start;
            if (r != 0)
            {
                sec_per_circ = ((r - 1) * sec_per_circ +
                                circ_time_diff.count()) / r;
            }
            else
            {
                sec_per_circ = circ_time_diff.count();
            }

            const double time_estimate = (realizations - r) * sec_per_circ;
            std::cout << "\rAverage circuits/second: "
                      << std::round((1 / sec_per_circ) * 100.0) / 100.0
                      << ", Realisations ETA: "
                      << static_cast<int>(time_estimate)
                      << "s               " << std::flush;
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved reduced density matrices to rho3_2d.bin.\n";
}


int main(int argc, char *argv[])
{
    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        std::cout << "Usage: " << argv[0]
                  << " [d = 1] [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0]\n";
        std::cout << "Description:\n";
        std::cout << "  d = dimensionality (1 or 2)\n";
        std::cout << "  n = number of qubits per dimension\n";
        std::cout << "  periods = number of periods (suggested 1D: n; 2D: 2*n^2)\n";
        std::cout << "  realizations = number of realizations (simulations per point)\n";
        std::cout << "  resolution = number of points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        return 0;
    }

    int d = (argc > 1) ? std::stoi(argv[1]) : 1;
    int n = (argc > 2) ? std::stoi(argv[2]) : ((d == 1) ? 10 : 4);
    int periods = (argc > 3) ? std::stoi(argv[3]) : ((d == 1) ? 10 : 32);
    int realizations = (argc > 4) ? std::stoi(argv[4]) : 10;
    int res = (argc > 5) ? std::stoi(argv[5]) : 5;
    double p_min = (argc > 6) ? std::stod(argv[6]) : 0.0;
    double p_max = (argc > 7) ? std::stod(argv[7]) : 1.0;

    auto start = std::chrono::steady_clock::now(); // Get start time

    if (d == 1)
    {
        std::cout << "Running 1D " << n <<"-qubit simulation of " << realizations*res << " circuits.\n";
        run_1d(n, periods, realizations, res, p_min, p_max);
    }
    else if (d == 2)
    {
        std::cout << "Running 2D " << n << "x" << n << " = " << n*n << "-qubit simulation of " << realizations*res << " circuits.\n";
        run_2d(n, n, periods, realizations, res, p_min, p_max);
    }

    auto end = std::chrono::steady_clock::now(); // Get end time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
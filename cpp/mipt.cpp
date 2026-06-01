#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "gmn.hpp"
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
#include <cstring>

#ifdef MIPT_ENABLE_CUDA_RHO
#include "rho_reduce.hpp"
#endif

using Complex = std::complex<double>;
using MatrixXc = Eigen::MatrixXcd;
using VectorXc = Eigen::VectorXcd;


// Keep your existing Eigen typedefs/includes if these are already defined:
// using VectorXc = Eigen::VectorXcd;
// using MatrixXc = Eigen::MatrixXcd;

namespace
{
    constexpr int GMN_KEEP_QUBITS = 3;
    constexpr int GMN_KEEP_DIM = 1 << GMN_KEEP_QUBITS; // 8
    constexpr int GMN_RHO_SIZE = GMN_KEEP_DIM * GMN_KEEP_DIM; // 64

    inline std::uint64_t checked_pow2_u64(int exponent)
    {
        if (exponent < 0 || exponent >= 63)
        {
            throw std::invalid_argument("Invalid qubit count: 2^n would overflow uint64_t.");
        }

        return std::uint64_t{1} << exponent;
    }

    inline void increment_low_bits(std::vector<int> &bits, int bit_count)
    {
        // Treat bits[0 : bit_count) as a little-endian binary counter.
        // Average cost is O(1), unlike rewriting every bit from an integer
        // on every loop iteration.
        for (int q = 0; q < bit_count; ++q)
        {
            bits[q] ^= 1;

            if (bits[q] != 0)
            {
                break;
            }
        }
    }

    inline void set_last3_bits(std::vector<int> &bits, int env_qubits, int abc)
    {
        // This matches your original indexing convention:
        //
        // bits[b] = (i >> b) & 1
        //
        // and your original kept-qubit order:
        //
        // kept = {n - 3, n - 2, n - 1}
        //
        // Therefore local ABC bit 0 maps to qubit n - 3,
        // local ABC bit 1 maps to qubit n - 2,
        // local ABC bit 2 maps to qubit n - 1.
        bits[env_qubits + 0] = (abc >> 0) & 1;
        bits[env_qubits + 1] = (abc >> 1) & 1;
        bits[env_qubits + 2] = (abc >> 2) & 1;
    }
}

// Direct replacement for:
//     cudaq_state_to_eigen(...)
//     partial_trace_statevector(...)
//     real-part extraction
//
// This computes only the final real 8x8 reduced density matrix needed by GMN.
//
// Assumption:
//     You are tracing out qubits 0 through n - 4 and keeping qubits
//     n - 3, n - 2, n - 1, exactly as in your posted code.
//
// Output:
//     rho_real is row-major, length 64.
//     rho_real[i * 8 + j] = real(rho_ABC(i,j)).
template <typename Real>
void last3_reduced_density_real_from_host_statevector(
    const std::complex<Real> *psi,
    int n,
    double *rho_real)
{
    if (psi == nullptr)
    {
        throw std::invalid_argument("state-vector pointer is null.");
    }

    std::fill(rho_real, rho_real + GMN_RHO_SIZE, 0.0);

    const int env_qubits = n - GMN_KEEP_QUBITS;
    const std::uint64_t env_dim = checked_pow2_u64(env_qubits);

    // CUDA-Q's state-vector indexing matches the old amplitude path used here:
    // qubit b maps to bit b in the linear state index. Therefore, for fixed
    // environment state `env` and kept subsystem basis `abc`, the full basis
    // index is env | (abc << env_qubits).
    for (std::uint64_t env = 0; env < env_dim; ++env)
    {
        std::array<std::complex<Real>, GMN_KEEP_DIM> amp{};

        for (int abc = 0; abc < GMN_KEEP_DIM; ++abc)
        {
            const std::uint64_t idx = env |
                (static_cast<std::uint64_t>(abc) << env_qubits);
            amp[abc] = psi[idx];
        }

        for (int i = 0; i < GMN_KEEP_DIM; ++i)
        {
            rho_real[i * GMN_KEEP_DIM + i] +=
                static_cast<double>(std::norm(amp[i]));

            for (int j = i + 1; j < GMN_KEEP_DIM; ++j)
            {
                const double re = static_cast<double>(
                    std::real(amp[i] * std::conj(amp[j])));

                rho_real[i * GMN_KEEP_DIM + j] += re;
                rho_real[j * GMN_KEEP_DIM + i] += re;
            }
        }
    }
}

void cudaq_last3_reduced_density_real(
    cudaq::state &state,
    int n,
    double *rho_real)
{
    if (rho_real == nullptr)
    {
        throw std::invalid_argument("rho_real pointer is null.");
    }

    if (n < GMN_KEEP_QUBITS)
    {
        throw std::invalid_argument("Need at least 3 qubits to keep the last 3 qubits.");
    }

    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dim = static_cast<std::size_t>(dim_u64);

    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();

    if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
    {
        throw std::runtime_error("Expected a contiguous rank-1 state-vector tensor.");
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        int status = 0;

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = mipt_cuda_last3_reduced_density_real_f64(tensor.data, n, rho_real);
        }
        else
        {
            status = mipt_cuda_last3_reduced_density_real_f32(tensor.data, n, rho_real);
        }

        if (status != 0)
        {
            throw std::runtime_error("CUDA last-3 reduced-density-matrix kernel failed with CUDA error code " +
                                     std::to_string(status));
        }

        return;
    }
#endif

    if (state.is_on_gpu())
    {
        // GPU target but no custom CUDA reduction linked. This is still much
        // faster than scalar amplitude() calls because it performs one bulk
        // transfer and then a direct contiguous host reduction.
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            std::vector<std::complex<double>> host_state(dim);
            state.to_host(host_state.data(), host_state.size());
            last3_reduced_density_real_from_host_statevector(
                host_state.data(), n, rho_real);
        }
        else
        {
            std::vector<std::complex<float>> host_state(dim);
            state.to_host(host_state.data(), host_state.size());
            last3_reduced_density_real_from_host_statevector(
                host_state.data(), n, rho_real);
        }

        return;
    }

    if (tensor.data == nullptr)
    {
        throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        last3_reduced_density_real_from_host_statevector(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n,
            rho_real);
    }
    else
    {
        last3_reduced_density_real_from_host_statevector(
            reinterpret_cast<const std::complex<float> *>(tensor.data),
            n,
            rho_real);
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

VectorXc cudaq_state_to_eigen(cudaq::state state, int n)
{
    int dim = 1 << n;
    VectorXc psi(dim);

    for (int i = 0; i < dim; ++i)
    {
        std::vector<int> bits(n);

        for (int b = 0; b < n; ++b)
            bits[b] = (i >> b) & 1;

        psi(i) = state.amplitude(bits);
    }

    return psi;
}

MatrixXc density_matrix(const VectorXc &psi)
{
    return psi * psi.adjoint();
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
    std::vector<double> ps = linspace(p_min, p_max, res);

    std::ofstream outfile("data.csv");
    std::ofstream infofile("info.csv");

    // headers
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << "," << p_min << "," << p_max << "\n";

    // headers
    outfile << "p,gmn\n";

    for (double p : ps)
    {
        std::cout << "Simulating p = "
                  << p << "...\n "
                  << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; r++)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0) start = std::chrono::steady_clock::now();

            auto layers = mipt_frontend(n, periods, p);
            
            if (r == 0) lap1 = std::chrono::steady_clock::now();

            // Get final statevector
            auto state =
                cudaq::get_state(MIPTKernel_1D{}, n, layers, true);

            if (r == 0) lap2 = std::chrono::steady_clock::now();

            std::array<double, 64> rho_real{};
            cudaq_last3_reduced_density_real(state, n, rho_real.data());

            if (r == 0) lap3 = std::chrono::steady_clock::now();

            double gmn = compute_gmn_mosek_real_8x8(rho_real.data());
            
            if (r == 0) lap4 = std::chrono::steady_clock::now();

            outfile << p << "," << gmn << "\n";

            if (r == 0) {
                std::chrono::duration<double> t_layer = lap1 - start;
                std::chrono::duration<double> t_qc = lap2 - lap1;
                std::chrono::duration<double> t_im = lap3 - lap2;
                std::chrono::duration<double> t_gmn = lap4 - lap3;
                std::cout << "Layer, QC, IM, GMN times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_im.count() << "s, "
                          << t_gmn.count() << "s\n";
            }
            auto circ_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> circ_time_diff = circ_time_end - circ_time_start;
            if (r != 0) 
            {
                sec_per_circ = ((r-1)*sec_per_circ + circ_time_diff.count())/r;
            }
            else 
            {
                sec_per_circ = circ_time_diff.count();
            }
            double time_estimate = (realizations - r)*sec_per_circ;
            std::cout << "\rAverage circuits/second: " << std::round((1 / sec_per_circ) * 100.0)/100.0 
                      << ", Realisations ETA: " << static_cast<int>(time_estimate) << "s               " << std::flush;
        }
        std::cout << "\n";
    }
}

void run_2d(int x, int y, int periods, int realizations, int res, double p_min, double p_max) 
{
    std::vector<double> ps = linspace(p_min, p_max, res);

    std::ofstream outfile("data2.csv");
    std::ofstream infofile("info2.csv");

    int n = x*y;

    // headers
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << "," << p_min << "," << p_max << "\n";

    // headers
    outfile << "p,gmn\n";

    for (double p : ps)
    {
        std::cout << "Simulating p = "
                  << p << "...\n "
                  << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; r++)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0) start = std::chrono::steady_clock::now();

            auto layers = mipt_frontend(n, periods, p);
            
            if (r == 0) lap1 = std::chrono::steady_clock::now();

            // Get final statevector
            auto state =
                cudaq::get_state(MIPTKernel_2D{}, x, y, layers, true);

            if (r == 0) lap2 = std::chrono::steady_clock::now();

            std::array<double, 64> rho_real{};
            cudaq_last3_reduced_density_real(state, n, rho_real.data());

            if (r == 0) lap3 = std::chrono::steady_clock::now();

            double gmn = compute_gmn_mosek_real_8x8(rho_real.data());
            
            if (r == 0) lap4 = std::chrono::steady_clock::now();

            outfile << p << "," << gmn << "\n";

            if (r == 0) {
                std::chrono::duration<double> t_layer = lap1 - start;
                std::chrono::duration<double> t_qc = lap2 - lap1;
                std::chrono::duration<double> t_im = lap3 - lap2;
                std::chrono::duration<double> t_gmn = lap4 - lap3;
                std::cout << "Layer, QC, IM, GMN times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_im.count() << "s, "
                          << t_gmn.count() << "s\n";
            }
            auto circ_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> circ_time_diff = circ_time_end - circ_time_start;
            if (r != 0) 
            {
                sec_per_circ = ((r-1)*sec_per_circ + circ_time_diff.count())/r;
            }
            else 
            {
                sec_per_circ = circ_time_diff.count();
            }
            double time_estimate = (realizations - r)*sec_per_circ;
            std::cout << "\rAverage circuits/second: " << std::round((1 / sec_per_circ) * 100.0)/100.0 
                      << ", Realisations ETA: " << static_cast<int>(time_estimate) << "s               " << std::flush;
        }
        std::cout << "\n";
    }
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
        std::cout << "Running 1D " << n <<"-qubit simulation of " << realizations*res << " circuits.\n"
        run_1d(n, periods, realizations, res, p_min, p_max);
    }
    else if (d == 2)
    {
        std::cout << "Running 2D " << n << "x" << n << " = " << n*n << "-qubit simulation of " << realizations*res << " circuits.\n"
        run_2d(n, n, periods, realizations, res, p_min, p_max);
    }

    auto end = std::chrono::steady_clock::now(); // Get end time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
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

using Complex = std::complex<double>;
using MatrixXc = Eigen::MatrixXcd;
using VectorXc = Eigen::VectorXcd;

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

MatrixXc partial_trace_statevector(const VectorXc &psi, int n, const std::vector<int> &traced_out)
{
    std::set<int> traced_set(traced_out.begin(), traced_out.end());

    std::vector<int> kept;
    for (int i = 0; i < n; ++i)
        if (!traced_set.count(i))
            kept.push_back(i);

    int dim_keep = 1 << kept.size();
    int dim_trace = 1 << traced_out.size();

    MatrixXc rho_red =
        MatrixXc::Zero(dim_keep, dim_keep);

    // iterate over all basis states of full system
    int dim_full = 1 << n;

    for (int a = 0; a < dim_full; ++a)
    {
        // decode indices into (kept, traced)
        int a_keep = 0;
        int a_trace = 0;

        for (size_t k = 0; k < kept.size(); ++k)
            a_keep |= (((a >> kept[k]) & 1) << k);

        for (size_t t = 0; t < traced_out.size(); ++t)
            a_trace |= (((a >> traced_out[t]) & 1) << t);

        for (int b = 0; b < dim_full; ++b)
        {
            int b_keep = 0;
            int b_trace = 0;

            for (size_t k = 0; k < kept.size(); ++k)
                b_keep |= (((b >> kept[k]) & 1) << k);

            for (size_t t = 0; t < traced_out.size(); ++t)
                b_trace |= (((b >> traced_out[t]) & 1) << t);

            // trace condition: traced subsystem must match
            if (a_trace != b_trace)
                continue;

            rho_red(a_keep, b_keep) +=
                psi(a) * std::conj(psi(b));
        }
    }

    return rho_red;
}

double entropy(const MatrixXc &rho)
{

    Eigen::SelfAdjointEigenSolver<MatrixXc>
        solver(rho);

    auto eigs = solver.eigenvalues();

    double S = 0.0;

    for (int i = 0; i < eigs.size(); ++i)
    {

        double lambda = eigs(i);

        if (lambda > 1e-12)
        {
            S -= lambda * std::log2(lambda);
        }
    }

    return S;
}

double tmi(const VectorXc &psi, int n)
{
    std::vector<int> notA, notB, notC;
    std::vector<int> notAB, notAC, notBC;

    for (int i = 0; i < n; ++i)
    {
        if (i != n - 3)
            notA.push_back(i);
        if (i != n - 2)
            notB.push_back(i);
        if (i != n - 1)
            notC.push_back(i);

        if (i != n - 3 && i != n - 2)
            notAB.push_back(i);
        if (i != n - 3 && i != n - 1)
            notAC.push_back(i);
        if (i != n - 2 && i != n - 1)
            notBC.push_back(i);
    }

    auto rho_A = partial_trace_statevector(psi, n, notA);
    auto rho_B = partial_trace_statevector(psi, n, notB);
    auto rho_C = partial_trace_statevector(psi, n, notC);

    auto rho_AB = partial_trace_statevector(psi, n, notAB);
    auto rho_AC = partial_trace_statevector(psi, n, notAC);
    auto rho_BC = partial_trace_statevector(psi, n, notBC);

    double S_A = entropy(rho_A);
    double S_B = entropy(rho_B);
    double S_C = entropy(rho_C);

    double S_AB = entropy(rho_AB);
    double S_AC = entropy(rho_AC);
    double S_BC = entropy(rho_BC);

    return S_A + S_B + S_C - S_AB - S_AC - S_BC;
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



void run(int n, int periods, int realizations, int res, double p_min, double p_max)
{
    std::vector<double> ps = linspace(p_min, p_max, res);

    std::ofstream outfile("data.csv");
    std::ofstream infofile("info.csv");

    // headers
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << "," << p_min << "," << p_max << "\n";

    // headers
    outfile << "p,tmi,gmn\n";

    for (double p : ps)
    {
        std::cout << "Simulating p = "
                  << p
                  << std::flush;
        for (int r = 0; r < realizations; r++)
        {
            auto layers = mipt_frontend(n, periods, p);

            // Get final statevector
            auto state =
                cudaq::get_state(MIPTKernel_1D{}, n, layers, true);

            // Convert to Eigen vector
            auto psi =
                cudaq_state_to_eigen(state, n);

            // Trace out all but last 3 qubits
            std::vector<int> traced;

            for (int i = 0; i < n - 3; ++i)
            {
                traced.push_back(i);
            }

            auto rho_ABC =
                partial_trace_statevector(psi, n, traced);

            std::array<double, 64> rho_real{};

            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    rho_real[i * 8 + j] = std::real(rho_ABC(i, j));
                }
            }

            double gmn = compute_gmn_mosek_real_8x8(rho_real.data());

            double I3 = tmi(psi, n);
            // std::cout << "TMI = "
            //           << I3
            //           << "\n";

            outfile << p << "," << I3 << "," << gmn << "\n";
            std::cout << "." << std::flush;
        }
        std::cout << "\n";
    }
}

int main(int argc, char *argv[])
{

    if (argv[1] == std::string("--help") || argv[1] == std::string("-h"))
    {
        std::cout << "Usage: " << argv[0]
                  << " [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0]\n";
        std::cout << "Description:\n";
        std::cout << "  n = number of qubits\n";
        std::cout << "  periods = number of periods (typically = n)\n";
        std::cout << "  realizations = number of realizations (simulations per point)\n";
        std::cout << "  resolution = number of points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        return 0;
    }
    int n = argv[1] ? std::stoi(argv[1]) : 10;
    int periods = argv[2] ? std::stoi(argv[2]) : 10;
    int realizations = argv[3] ? std::stoi(argv[3]) : 10;
    int res = argv[4] ? std::stoi(argv[4]) : 5;
    double p_min = argv[5] ? std::stod(argv[5]) : 0.0;
    double p_max = argv[6] ? std::stod(argv[6]) : 1.0;

    auto start = std::chrono::steady_clock::now(); // Get start time

    run(n, periods, realizations, res, p_min, p_max);

    auto end = std::chrono::steady_clock::now(); // Get end time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
#include <cudaq.h>
#include <vector>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <complex>
#include <cmath>
#include <set>
#include <fstream>
#include <chrono>
#include <cstdlib>

using Complex = std::complex<double>;
using MatrixXc = Eigen::MatrixXcd;
using VectorXc = Eigen::VectorXcd;

// Source - https://stackoverflow.com/a/27030598
// Posted by Akavall, modified by community. See post 'Timeline' for change history
// Retrieved 2026-05-25, License - CC BY-SA 3.0

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

double run_gmn_server(const MatrixXc &rho)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(50007);

    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr));

    int D = rho.rows();
    std::vector<double> buffer;
    buffer.reserve(D*D);

    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            buffer.push_back(std::real(rho(i,j)));

    send(sock, buffer.data(), buffer.size()*sizeof(double), 0);

    char reply[128] = {0};
    read(sock, reply, sizeof(reply));

    close(sock);

    return atof(reply);
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

MatrixXc partial_trace_statevector(
    const VectorXc &psi,
    int n,
    const std::vector<int> &traced_out)
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
        if (i != n - 3) notA.push_back(i);
        if (i != n - 2) notB.push_back(i);
        if (i != n - 1) notC.push_back(i);

        if (i != n - 3 && i != n - 2) notAB.push_back(i);
        if (i != n - 3 && i != n - 1) notAC.push_back(i);
        if (i != n - 2 && i != n - 1) notBC.push_back(i);
    }

    auto rho_A  = partial_trace_statevector(psi, n, notA);
    auto rho_B  = partial_trace_statevector(psi, n, notB);
    auto rho_C  = partial_trace_statevector(psi, n, notC);

    auto rho_AB = partial_trace_statevector(psi, n, notAB);
    auto rho_AC = partial_trace_statevector(psi, n, notAC);
    auto rho_BC = partial_trace_statevector(psi, n, notBC);

    double S_A  = entropy(rho_A);
    double S_B  = entropy(rho_B);
    double S_C  = entropy(rho_C);

    double S_AB = entropy(rho_AB);
    double S_AC = entropy(rho_AC);
    double S_BC = entropy(rho_BC);

    return S_A + S_B + S_C
         - S_AB - S_AC - S_BC;
}

std::vector<double> matrix_to_vec(const MatrixXc &rho)
{
    int D = rho.rows();
    std::vector<double> v(D * D);

    for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
    {
        // GMN uses real matrix (as in your code)
        v[i * D + j] = std::real(rho(i,j));
    }

    return v;
}

struct LayerData
{
    std::vector<int> measure_flags;
    std::vector<double> theta_x;
    std::vector<double> theta_y;
    std::vector<double> theta_z;
};

struct MIPTKernel
{

    void operator()(int n,
                    const std::vector<LayerData> &layers) __qpu__
    {

        cudaq::qvector q(n);

        for (std::size_t layer = 0;
             layer < layers.size();
             ++layer)
        {

            bool even = (layer % 2 == 0);

            int start = even ? 0 : 1;

            // Two-qubit brickwork layer
            for (int i = start; i < n - 1; i += 2)
            {

                // Random local rotations
                rx(layers[layer].theta_x[i], q[i]);
                ry(layers[layer].theta_y[i], q[i]);
                rz(layers[layer].theta_z[i], q[i]);

                rx(layers[layer].theta_x[i + 1], q[i + 1]);
                ry(layers[layer].theta_y[i + 1], q[i + 1]);
                rz(layers[layer].theta_z[i + 1], q[i + 1]);

                // Entangling gate
                cx(q[i], q[i + 1]);
            }

            // Mid-circuit measurements
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

std::vector<LayerData> mipt_frontend(int n, int depth, double p)
{
    std::mt19937 rng(std::random_device{}());

    std::uniform_real_distribution<double>
        angle_dist(0.0, 2.0 * M_PI);

    std::uniform_real_distribution<double>
        prob_dist(0.0, 1.0);

    std::vector<LayerData> layers;

    // Generate random circuit CLASSICALLY
    int eff_depth = depth;
    // 50% chance of an extra layer
    if (prob_dist(rng) < 0.5) {
        eff_depth += 1; 
    }
    for (int d = 0; d < eff_depth; ++d)
    {

        LayerData layer;

        layer.measure_flags.resize(n);
        layer.theta_x.resize(n);
        layer.theta_y.resize(n);
        layer.theta_z.resize(n);

        for (int q = 0; q < n; ++q)
        {

            layer.theta_x[q] = angle_dist(rng);
            layer.theta_y[q] = angle_dist(rng);
            layer.theta_z[q] = angle_dist(rng);

            layer.measure_flags[q] =
                (prob_dist(rng) < p);
        }

        layers.push_back(layer);
    }

    return layers;
}

void run(int n, int depth, int realizations, int res, double p_min, double p_max)
{
    std::vector<double> ps = linspace(p_min, p_max, res);

    std::ofstream outfile("data.csv");

    // headers
    outfile << "p,tmi,gmn\n";

    for (double p : ps)
    {
        std::cout << "Simulating p = "
                  << p
                  << std::flush;
        for (int r = 0; r < realizations; r++)
        {
            auto layers = mipt_frontend(n, depth, p);

            auto counts = cudaq::sample(
                MIPTKernel{},
                n,
                layers,
                4096
            );

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

            auto rho_vec = matrix_to_vec(rho_ABC);

            double gmn = run_gmn_server(rho_ABC);

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

int main(int argc, char* argv[])
{

    int n = argv[1] ? std::stoi(argv[1]) : 10;
    int depth = argv[2] ? std::stoi(argv[2]) : 20;
    int realizations = argv[3] ? std::stoi(argv[3]) : 10;
    int res = argv[4] ? std::stoi(argv[4]) : 5;
    double p_min = argv[5] ? std::stod(argv[5]) : 0.0;
    double p_max = argv[6] ? std::stod(argv[6]) : 1.0;

    auto start = std::chrono::steady_clock::now(); // Get start time

    run(n, depth, realizations, res, p_min, p_max);

    auto end = std::chrono::steady_clock::now();   // Get end time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
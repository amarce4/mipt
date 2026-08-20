// Validation for the even-parity-sector encoding (mipt/cusv/parity.hpp).
//
// The encoding is exact, not approximate, so it is pinned against a
// from-first-principles host reference rather than against a tolerance: the
// reference applies the *unencoded* operator list to a full 2^n state vector
// in fp64 and projects it with whatever measurement outcomes the engine drew.
// Deliberately not a transcription of the encoder -- it never forms an encoded
// index -- so the two agree only if the rewriting is actually right.
//
//   Part 1  Parity confinement: on the production layer builders, the
//           odd-parity half of the full-state reference is exactly zero at
//           every layer. This is the premise the whole encoding rests on.
//   Part 2  Unitary equivalence at p = 0: covers plain bonds on the encoded
//           index, the boundary-bond reduction, the Jordan-Wigner fold, and
//           the in-place expansion.
//   Part 3  Measured trajectories, compared layer by layer with the host
//           reference forced to follow the engine's own outcomes. Covers the
//           split joint measurement, including the parity chunk for site n-1.
//   Part 4  The parity measurement primitives against direct host sums.
//
// Usage: ./cusv_parity_tests [n] [periods] [seed]

#include "mipt/circuit.hpp"
#include "mipt/cusv/engine.hpp"
#include "mipt/cusv/parity.hpp"

#include <cudaq.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

using Complex = std::complex<double>;

// See dump_bond.cpp and cusv_equiv.cpp: nvq++ emits a registration reference to
// every __qpu__ inline free function pulled in from a header, so they must be
// ODR-used here even though this test drives the engine directly.
struct KeepLayerHelpersAlive
{
    void operator()(int n, const std::vector<mipt::MmsLayer> &mms,
                    const std::vector<mipt::HaarLayer> &haar,
                    const std::vector<mipt::RppuLayer> &rppu,
                    const std::vector<mipt::RfgsLayer> &rfgs) __qpu__
    {
        cudaq::qvector q(n);
        mipt::apply_mms_layers(q, n, mms, true);
        mipt::apply_haar_layers(q, n, haar);
        mipt::apply_rppu_layers(q, n, rppu);
        mipt::apply_rfgs_layers(q, n, rfgs, true);
    }
};

int failures = 0;

void check(bool ok, const std::string &what)
{
    if (!ok)
    {
        std::cout << "  FAIL: " << what << "\n";
        ++failures;
    }
}

// ------------------------------------------------------- host reference

// Applies one operator of the *unencoded* list to a full 2^n state vector.
void apply_op_host(std::vector<Complex> &psi, const mipt::cusv::Op &op)
{
    if (op.kind == mipt::cusv::OpKind::JwString)
    {
        for (std::size_t i = 0; i < psi.size(); ++i)
        {
            if (((i >> op.string_control) & 1u) != 0u &&
                (__builtin_popcountll(i & op.string_mask) & 1) != 0)
            {
                psi[i] = -psi[i];
            }
        }
        return;
    }
    if (op.kind != mipt::cusv::OpKind::Matrix)
    {
        std::cout << "  FAIL: host reference saw an encoded operator\n";
        ++failures;
        return;
    }
    const int k = op.matrix.targets;
    const int dim = 1 << k;
    std::vector<Complex> in(dim), out(dim);
    std::vector<std::size_t> stride(k);
    std::size_t target_mask = 0;
    for (int b = 0; b < k; ++b)
    {
        stride[b] = std::size_t{1} << op.targets[b];
        target_mask |= stride[b];
    }
    for (std::size_t base = 0; base < psi.size(); ++base)
    {
        if ((base & target_mask) != 0u) { continue; }
        for (int r = 0; r < dim; ++r)
        {
            std::size_t idx = base;
            for (int b = 0; b < k; ++b) { if ((r >> b) & 1) { idx |= stride[b]; } }
            in[r] = psi[idx];
        }
        for (int r = 0; r < dim; ++r)
        {
            Complex s{0.0, 0.0};
            for (int c = 0; c < dim; ++c) { s += op.matrix.values[r * dim + c] * in[c]; }
            out[r] = s;
        }
        for (int r = 0; r < dim; ++r)
        {
            std::size_t idx = base;
            for (int b = 0; b < k; ++b) { if ((r >> b) & 1) { idx |= stride[b]; } }
            psi[idx] = out[r];
        }
    }
}

// Projects onto `site == bit` and renormalises.
void project_host(std::vector<Complex> &psi, int site, int bit)
{
    double weight = 0.0;
    for (std::size_t i = 0; i < psi.size(); ++i)
    {
        if (static_cast<int>((i >> site) & 1u) == bit) { weight += std::norm(psi[i]); }
    }
    const double scale = 1.0 / std::sqrt(weight);
    for (std::size_t i = 0; i < psi.size(); ++i)
    {
        psi[i] = (static_cast<int>((i >> site) & 1u) == bit) ? psi[i] * scale : Complex{0.0, 0.0};
    }
}

double largest_odd_parity_amplitude(const std::vector<Complex> &psi)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < psi.size(); ++i)
    {
        if ((__builtin_popcountll(i) & 1) != 0) { worst = std::max(worst, std::abs(psi[i])); }
    }
    return worst;
}

double max_deviation(const std::vector<Complex> &a, const std::vector<Complex> &b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { worst = std::max(worst, std::abs(a[i] - b[i])); }
    return worst;
}

// ------------------------------------------------------- device helpers

struct DeviceState
{
    void *data = nullptr;
    bool fp64 = false;
    int n = 0;

    DeviceState(int qubits, bool use_fp64) : fp64(use_fp64), n(qubits)
    {
        const std::size_t bytes = (std::size_t{1} << qubits) * (use_fp64 ? 16u : 8u);
        if (cudaMalloc(&data, bytes) != cudaSuccess) { data = nullptr; }
    }
    ~DeviceState() { if (data != nullptr) { cudaFree(data); } }
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;

    void upload(const std::vector<Complex> &host) const
    {
        if (fp64)
        {
            cudaMemcpy(data, host.data(), host.size() * 16u, cudaMemcpyHostToDevice);
            return;
        }
        std::vector<std::complex<float>> narrow(host.size());
        std::transform(host.begin(), host.end(), narrow.begin(), [](const Complex &z)
                       { return std::complex<float>(static_cast<float>(z.real()),
                                                    static_cast<float>(z.imag())); });
        cudaMemcpy(data, narrow.data(), narrow.size() * 8u, cudaMemcpyHostToDevice);
    }

    std::vector<Complex> download(std::size_t elements) const
    {
        std::vector<Complex> host(elements);
        if (fp64)
        {
            cudaMemcpy(host.data(), data, elements * 16u, cudaMemcpyDeviceToHost);
            return host;
        }
        std::vector<std::complex<float>> narrow(elements);
        cudaMemcpy(narrow.data(), data, elements * 8u, cudaMemcpyDeviceToHost);
        std::transform(narrow.begin(), narrow.end(), host.begin(),
                       [](const std::complex<float> &z) { return Complex(z.real(), z.imag()); });
        return host;
    }
};

// The full 2^n vector an encoded 2^(n-1) one stands for.
std::vector<Complex> expand_host(const std::vector<Complex> &encoded, int n)
{
    std::vector<Complex> full(std::size_t{1} << n, Complex{0.0, 0.0});
    const std::size_t half = std::size_t{1} << (n - 1);
    for (std::size_t e = 0; e < half; ++e)
    {
        const std::size_t derived = (__builtin_popcountll(e) & 1) != 0 ? half : 0u;
        full[e | derived] = encoded[e];
    }
    return full;
}

std::vector<mipt::RppuLayer> make_layers(int n, int periods, double p, std::mt19937 &rng)
{
    std::vector<mipt::RppuLayer> layers;
    mipt::build_rppu_layers_with_boundary(layers, n, periods, p, true, rng,
                                          mipt::RppuBoundaryMode::FermionicJWString);
    return layers;
}

// ------------------------------------------------------------ the parts

// Part 1 + 2: the premise, and unitary equivalence at p = 0.
void test_unitary(int n, int periods, unsigned seed, bool fp64, double tolerance)
{
    std::mt19937 rng(seed);
    const auto layers = make_layers(n, periods, 0.0, rng);

    std::vector<Complex> reference(std::size_t{1} << n, Complex{0.0, 0.0});
    reference[0] = Complex{1.0, 0.0};
    double worst_odd = 0.0;
    std::vector<std::vector<mipt::cusv::Op>> raw;
    for (const auto &layer : layers)
    {
        auto ops = mipt::cusv::build_rppu_layer_ops(layer, n);
        for (const auto &op : ops.ops) { apply_op_host(reference, op); }
        worst_odd = std::max(worst_odd, largest_odd_parity_amplitude(reference));
        raw.push_back(std::move(ops.ops));
    }
    check(worst_odd == 0.0,
          "odd-parity half is exactly zero at every layer (saw " + std::to_string(worst_odd) + ")");

    DeviceState device(n, fp64);
    check(device.data != nullptr, "device allocation");
    if (device.data == nullptr) { return; }

    std::vector<Complex> start(std::size_t{1} << n, Complex{0.0, 0.0});
    start[0] = Complex{1.0, 0.0};
    device.upload(start);

    mipt::cusv::Engine engine(fp64);
    engine.adopt(device.data, n - 1);
    const int block = mipt::cusv::max_block_targets();
    for (const auto &ops : raw)
    {
        auto encoded = mipt::cusv::encode_parity_sector(ops, n);
        check(encoded.has_value(), "layer is encodable");
        if (!encoded) { return; }
        engine.apply_ops(*encoded, block);
    }
    engine.expand_parity_sector(n);
    cudaDeviceSynchronize();

    const auto got = device.download(std::size_t{1} << n);
    const double deviation = max_deviation(got, reference);
    std::printf("  n=%2d periods=%2d %s  unitary: max |encoded - full| = %.3e\n", n, periods,
                fp64 ? "fp64" : "fp32", deviation);
    check(deviation < tolerance, "unitary equivalence within tolerance");
}

// Part 3: measured trajectories, layer by layer, with the reference forced to
// follow the outcomes the engine drew.
void test_measured(int n, int periods, double p, unsigned seed, bool fp64, double tolerance)
{
    std::mt19937 layer_rng(seed);
    const auto layers = make_layers(n, periods, p, layer_rng);

    DeviceState device(n, fp64);
    check(device.data != nullptr, "device allocation");
    if (device.data == nullptr) { return; }

    std::vector<Complex> start(std::size_t{1} << n, Complex{0.0, 0.0});
    start[0] = Complex{1.0, 0.0};
    device.upload(start);

    std::vector<Complex> reference = start;
    mipt::cusv::Engine engine(fp64);
    engine.adopt(device.data, n - 1);
    std::mt19937 measure_rng(seed ^ 0x5bd1u);

    const int block = mipt::cusv::max_block_targets();
    double worst = 0.0;
    std::size_t derived_measurements = 0;
    for (const auto &layer : layers)
    {
        const auto ops = mipt::cusv::build_rppu_layer_ops(layer, n);
        auto encoded = mipt::cusv::encode_parity_sector(ops.ops, n);
        check(encoded.has_value(), "layer is encodable");
        if (!encoded) { return; }

        engine.apply_ops(*encoded, block);
        std::vector<int> outcomes;
        engine.measure_layer_encoded(ops.measure_sites, n, measure_rng, &outcomes);
        cudaDeviceSynchronize();

        for (const auto &op : ops.ops) { apply_op_host(reference, op); }
        for (std::size_t k = 0; k < ops.measure_sites.size(); ++k)
        {
            project_host(reference, ops.measure_sites[k], outcomes[k]);
            if (ops.measure_sites[k] == n - 1) { ++derived_measurements; }
        }

        const auto encoded_state = device.download(std::size_t{1} << (n - 1));
        worst = std::max(worst, max_deviation(expand_host(encoded_state, n), reference));
    }
    std::printf("  n=%2d periods=%2d p=%.2f %s  measured: max |encoded - full| = %.3e over all "
                "layers (%zu parity measurements)\n",
                n, periods, p, fp64 ? "fp64" : "fp32", worst, derived_measurements);
    check(worst < tolerance, "measured equivalence within tolerance");
    check(derived_measurements > 0, "site n-1 was measured at least once");
}

// Part 4: the parity measurement primitives against direct host sums.
void test_parity_primitives(int n, unsigned seed, bool fp64, double tolerance)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const std::size_t half = std::size_t{1} << (n - 1);

    std::vector<Complex> encoded(half);
    double norm = 0.0;
    for (auto &z : encoded)
    {
        z = Complex{gauss(rng), gauss(rng)};
        norm += std::norm(z);
    }
    for (auto &z : encoded) { z /= std::sqrt(norm); }

    std::vector<Complex> buffer(std::size_t{1} << n, Complex{0.0, 0.0});
    std::copy(encoded.begin(), encoded.end(), buffer.begin());

    DeviceState device(n, fp64);
    if (device.data == nullptr) { return; }
    device.upload(buffer);

    // Weights: parity(e) is exactly b_{n-1}, so the host sums split the same way.
    double expected[2] = {0.0, 0.0};
    for (std::size_t e = 0; e < half; ++e)
    {
        expected[(__builtin_popcountll(e) & 1) != 0 ? 1 : 0] += std::norm(encoded[e]);
    }

    double weights[2] = {0.0, 0.0};
    const int status = fp64 ? mipt_cuda_parity_abs2_f64(device.data, n - 1, weights)
                            : mipt_cuda_parity_abs2_f32(device.data, n - 1, weights);
    check(status == 0, "parity abs2 launch");
    const double weight_error = std::max(std::abs(weights[0] - expected[0]),
                                         std::abs(weights[1] - expected[1]));
    std::printf("  n=%2d %s  parity weights: {%.6f, %.6f}, max error %.3e\n", n,
                fp64 ? "fp64" : "fp32", weights[0], weights[1], weight_error);
    check(weight_error < tolerance, "parity weights match a direct host sum");

    for (int outcome = 0; outcome < 2; ++outcome)
    {
        device.upload(buffer);
        const int collapse =
            fp64 ? mipt_cuda_parity_collapse_f64(device.data, n - 1, outcome, expected[outcome])
                 : mipt_cuda_parity_collapse_f32(device.data, n - 1, outcome, expected[outcome]);
        check(collapse == 0, "parity collapse launch");
        cudaDeviceSynchronize();

        std::vector<Complex> want(half, Complex{0.0, 0.0});
        const double scale = 1.0 / std::sqrt(expected[outcome]);
        for (std::size_t e = 0; e < half; ++e)
        {
            if (((__builtin_popcountll(e) & 1) != 0 ? 1 : 0) == outcome)
            {
                want[e] = encoded[e] * scale;
            }
        }
        const auto got = device.download(half);
        const double deviation = max_deviation(got, want);
        check(deviation < tolerance,
              "parity collapse onto " + std::to_string(outcome) + " matches the host projection");
        if (deviation >= tolerance)
        {
            std::printf("    collapse(%d) deviation %.3e\n", outcome, deviation);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 12;
    const int periods = argc > 2 ? std::atoi(argv[2]) : 6;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 2026u;

    if (argc > 99)
    {
        cudaq::get_state(KeepLayerHelpersAlive{}, 2, std::vector<mipt::MmsLayer>{},
                         std::vector<mipt::HaarLayer>{}, std::vector<mipt::RppuLayer>{},
                         std::vector<mipt::RfgsLayer>{});
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        std::cout << "No CUDA device; skipping parity-sector tests.\n";
        return 0;
    }

    std::cout << "Parity-sector encoding tests (n=" << n << ", periods=" << periods
              << ", seed=" << seed << ")\n";

    // fp32 is the production configuration; fp64 pins the algebra itself.
    test_unitary(n, periods, seed, /*fp64=*/false, 1e-5);
    test_unitary(n, periods, seed, /*fp64=*/true, 1e-12);
    test_unitary(n - 2, periods + 2, seed + 1u, /*fp64=*/true, 1e-12);

    test_measured(n, periods, 0.15, seed, /*fp64=*/false, 1e-5);
    test_measured(n, periods, 0.15, seed, /*fp64=*/true, 1e-12);
    test_measured(n, periods, 0.45, seed + 7u, /*fp64=*/true, 1e-12);

    test_parity_primitives(n, seed + 3u, /*fp64=*/false, 1e-6);
    test_parity_primitives(n, seed + 3u, /*fp64=*/true, 1e-14);

    if (failures != 0)
    {
        std::cout << failures << " parity-sector check(s) FAILED.\n";
        return 1;
    }
    std::cout << "All parity-sector encoding tests passed.\n";
    return 0;
}

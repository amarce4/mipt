// End-to-end equivalence harness: CUDA-Q reference vs the cuStateVec engine.
//
// Part 1 (unitary): builds a real brickwork circuit at p=0 so the evolution is
// deterministic, runs it both ways, and compares state vectors amplitude by
// amplitude. This covers optimization #1 (fused bond matrices), #2 (JW string
// as one diagonal pass), #4 (persistent buffer) and #5 (Kronecker blocking).
//
// Part 2 (measurement): checks optimization #3 against a host reference --
// that the joint distribution, the collapsed state, and the returned surprisal
// all match sequential single-site Born sampling with renormalisation.
//
// Usage: ./cusv_equiv [n] [timesteps] [seed]

#include "mipt/circuit.hpp"
#include "mipt/cusv/engine.hpp"
#include "mipt/probed.hpp"

#include <cudaq.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void check(bool ok, const std::string &what)
{
    if (!ok)
    {
        std::cout << "  FAIL: " << what << "\n";
        ++failures;
    }
}

// See dump_bond.cpp: nvq++ emits references to every __qpu__ inline free
// function pulled in from a header, so they must be ODR-used here.
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

// cudaq::state::to_host refuses a complex<double> buffer on an fp32 build, so
// read at the build precision and widen.
void reference_to_host(cudaq::state &state, std::size_t dim,
                       std::vector<std::complex<double>> &out)
{
    out.resize(dim);
#if MIPT_CUDAQ_PRECISION == 64
    state.to_host(out.data(), out.size());
#else
    std::vector<std::complex<float>> narrow(dim);
    state.to_host(narrow.data(), narrow.size());
    std::transform(narrow.begin(), narrow.end(), out.begin(),
                   [](const std::complex<float> &z)
                   { return std::complex<double>(z.real(), z.imag()); });
#endif
}

struct Comparison
{
    double max_abs = 0.0;
    double overlap = 0.0;
};

Comparison compare(const std::vector<std::complex<double>> &a,
                   const std::vector<std::complex<double>> &b)
{
    Comparison out;
    std::complex<double> inner{0.0, 0.0};
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        out.max_abs = std::max(out.max_abs, std::abs(a[i] - b[i]));
        inner += std::conj(a[i]) * b[i];
    }
    out.overlap = std::abs(inner);
    return out;
}

// ---------------------------------------------------------------- part 1

void run_unitary_case(mipt::CircuitType type, int n, int timesteps, unsigned seed, bool fp64,
                      int block_targets)
{
    const char *name = mipt::circuit_type_name(type).data();

    std::vector<std::complex<double>> reference;
    std::vector<mipt::cusv::LayerOps> ops_per_layer;

    // Build layers with the production samplers at p=0, then run both engines
    // over exactly the same layer objects.
    switch (type)
    {
    case mipt::CircuitType::MMS:
    {
        std::mt19937 rng(seed);
        std::vector<mipt::MmsLayer> layers(static_cast<std::size_t>(timesteps));
        for (int s = 0; s < timesteps; ++s)
        {
            mipt::fill_mms_layer(layers[static_cast<std::size_t>(s)], n, s % 2, 0.0, rng);
        }
        auto state = cudaq::get_state(mipt::MmsKernel1D{}, n, layers, true);
        reference_to_host(state, std::size_t{1} << n, reference);
        for (const auto &layer : layers)
        {
            ops_per_layer.push_back(mipt::cusv::build_mms_layer_ops(layer, n, true));
        }
        break;
    }
    case mipt::CircuitType::Haar:
    {
        std::mt19937 rng(seed);
        std::vector<mipt::HaarLayer> layers(static_cast<std::size_t>(timesteps));
        for (int s = 0; s < timesteps; ++s)
        {
            mipt::fill_haar_layer(layers[static_cast<std::size_t>(s)], n, s % 2, 0.0, true, rng);
        }
        auto state = cudaq::get_state(mipt::HaarKernel1D{}, n, layers);
        reference_to_host(state, std::size_t{1} << n, reference);
        for (const auto &layer : layers)
        {
            ops_per_layer.push_back(mipt::cusv::build_haar_layer_ops(layer, n));
        }
        break;
    }
    case mipt::CircuitType::FermionRPPU:
    case mipt::CircuitType::QubitRPPU:
    {
        const bool qubit_boundary = type == mipt::CircuitType::QubitRPPU;
        const mipt::RppuBoundaryMode mode = qubit_boundary
                                                ? mipt::RppuBoundaryMode::QubitDirect
                                                : mipt::RppuBoundaryMode::FermionicJWString;
        std::mt19937 rng(seed);
        std::vector<mipt::RppuLayer> layers(static_cast<std::size_t>(timesteps));
        for (int s = 0; s < timesteps; ++s)
        {
            mipt::fill_rppu_layer(layers[static_cast<std::size_t>(s)], n, (s % 2) != 0, 0.0, true,
                                  rng, mode);
        }
        auto state = cudaq::get_state(mipt::RppuKernel1D{}, n, layers);
        reference_to_host(state, std::size_t{1} << n, reference);
        for (const auto &layer : layers)
        {
            ops_per_layer.push_back(mipt::cusv::build_rppu_layer_ops(layer, n));
        }
        break;
    }
    default:
        return;
    }

    mipt::cusv::Engine engine(fp64);
    engine.reset(n);
    std::size_t total_ops = 0;
    std::size_t blocked_ops = 0;
    for (const auto &layer : ops_per_layer)
    {
        total_ops += layer.ops.size();
        blocked_ops += mipt::cusv::block_disjoint_ops(layer.ops, block_targets).size();
        engine.apply_ops(layer.ops, block_targets);
    }
    engine.synchronize();

    std::vector<std::complex<double>> mine;
    engine.to_host(mine);

    const auto c = compare(reference, mine);
    const double norm = engine.squared_norm();
    std::printf("  %-28s n=%2d t=%2d %s block=%d : max|diff|=%.3e  |<ref|new>|=%.12f  norm=%.9f\n",
                name, n, timesteps, fp64 ? "fp64" : "fp32", block_targets, c.max_abs, c.overlap,
                norm);
    std::printf("      ops: %zu -> %zu after blocking\n", total_ops, blocked_ops);

    const double tol = fp64 ? 1e-11 : 2e-4;
    check(c.max_abs < tol, std::string(name) + " statevector matches CUDA-Q");
    check(std::abs(c.overlap - 1.0) < (fp64 ? 1e-12 : 1e-5), std::string(name) + " overlap is 1");
    check(std::abs(norm - 1.0) < (fp64 ? 1e-12 : 1e-5), std::string(name) + " norm preserved");
}

// ---------------------------------------------------------------- part 2

// Host reference: sequential single-site Born sampling with renormalisation,
// exactly what free_energy.exe does today, accumulating per-site surprisals.
double host_sequential_collapse(std::vector<std::complex<double>> &psi, int n,
                                const std::vector<int> &sites, const std::vector<int> &outcomes)
{
    double surprisal = 0.0;
    for (std::size_t k = 0; k < sites.size(); ++k)
    {
        const std::uint64_t mask = std::uint64_t{1} << sites[k];
        double total = 0.0;
        double branch = 0.0;
        for (std::uint64_t i = 0; i < psi.size(); ++i)
        {
            const double w = std::norm(psi[i]);
            total += w;
            const int bit = (i & mask) != 0 ? 1 : 0;
            if (bit == outcomes[k]) { branch += w; }
        }
        surprisal += -std::log(branch / total);
        const double scale = 1.0 / std::sqrt(branch);
        for (std::uint64_t i = 0; i < psi.size(); ++i)
        {
            const int bit = (i & mask) != 0 ? 1 : 0;
            psi[i] = (bit == outcomes[k]) ? psi[i] * scale : std::complex<double>{0.0, 0.0};
        }
    }
    return surprisal;
}

void run_measurement_case(int n, unsigned seed, bool fp64)
{
    // Prepare a generic entangled state with a few Haar layers.
    std::mt19937 rng(seed);
    const int prep = 4;
    std::vector<mipt::HaarLayer> layers(static_cast<std::size_t>(prep));
    for (int s = 0; s < prep; ++s)
    {
        mipt::fill_haar_layer(layers[static_cast<std::size_t>(s)], n, s % 2, 0.0, true, rng);
    }

    mipt::cusv::Engine engine(fp64);
    engine.reset(n);
    for (const auto &layer : layers)
    {
        engine.apply_ops(mipt::cusv::build_haar_layer_ops(layer, n).ops, 6);
    }
    std::vector<std::complex<double>> before;
    engine.to_host(before);

    // Measure a set of sites jointly.
    std::vector<int> sites;
    for (int s = 1; s < n; s += 2) { sites.push_back(s); }

    std::mt19937 draw_rng(seed + 1000u);
    std::vector<int> outcomes;
    const double batched_surprisal = engine.measure_layer(sites, draw_rng, &outcomes);
    engine.synchronize();

    std::vector<std::complex<double>> after;
    engine.to_host(after);

    // Host reference driven by the *same* sampled outcomes.
    std::vector<std::complex<double>> host = before;
    const double sequential_surprisal = host_sequential_collapse(host, n, sites, outcomes);

    const auto c = compare(host, after);
    std::printf("  measurement  n=%2d %s sites=%zu : max|diff|=%.3e  surprisal batched=%.12f "
                "sequential=%.12f  d=%.3e\n",
                n, fp64 ? "fp64" : "fp32", sites.size(), c.max_abs, batched_surprisal,
                sequential_surprisal, std::abs(batched_surprisal - sequential_surprisal));

    const double tol = fp64 ? 1e-11 : 2e-4;
    check(c.max_abs < tol, "batched collapse matches sequential collapse");
    check(std::abs(batched_surprisal - sequential_surprisal) < (fp64 ? 1e-10 : 1e-3),
          "joint surprisal equals summed sequential conditionals");
    check(std::abs(engine.squared_norm() - 1.0) < (fp64 ? 1e-12 : 1e-5),
          "state renormalised after collapse");
}

// Statistical check: the sampled outcome frequencies must follow the exact
// joint Born distribution.
void run_sampling_statistics(int n, unsigned seed, bool fp64)
{
    std::mt19937 rng(seed);
    std::vector<mipt::HaarLayer> layers(3);
    for (int s = 0; s < 3; ++s)
    {
        mipt::fill_haar_layer(layers[static_cast<std::size_t>(s)], n, s % 2, 0.0, true, rng);
    }

    const std::vector<int> sites{0, 1, 2};
    const std::size_t bins = 8;

    // Exact joint distribution from a host state.
    mipt::cusv::Engine engine(fp64);
    engine.reset(n);
    for (const auto &layer : layers)
    {
        engine.apply_ops(mipt::cusv::build_haar_layer_ops(layer, n).ops, 6);
    }
    std::vector<std::complex<double>> psi;
    engine.to_host(psi);
    std::vector<double> exact(bins, 0.0);
    for (std::uint64_t i = 0; i < psi.size(); ++i)
    {
        std::size_t bin = 0;
        for (std::size_t k = 0; k < sites.size(); ++k)
        {
            if ((i >> sites[k]) & 1u) { bin |= std::size_t{1} << k; }
        }
        exact[bin] += std::norm(psi[i]);
    }

    const int trials = 40000;
    std::vector<double> counts(bins, 0.0);
    std::mt19937 draw_rng(seed + 77u);
    std::vector<int> outcomes;
    for (int t = 0; t < trials; ++t)
    {
        engine.set_from_host(psi, n);
        engine.measure_layer(sites, draw_rng, &outcomes);
        std::size_t bin = 0;
        for (std::size_t k = 0; k < outcomes.size(); ++k)
        {
            if (outcomes[k]) { bin |= std::size_t{1} << k; }
        }
        counts[bin] += 1.0;
    }

    double worst_z = 0.0;
    for (std::size_t b = 0; b < bins; ++b)
    {
        const double p = exact[b];
        const double expected = p * trials;
        const double sigma = std::sqrt(std::max(1.0, expected * (1.0 - p)));
        worst_z = std::max(worst_z, std::abs(counts[b] - expected) / sigma);
    }
    std::printf("  sampling stats n=%2d %s trials=%d : worst |z| over 8 joint bins = %.2f\n", n,
                fp64 ? "fp64" : "fp32", trials, worst_z);
    check(worst_z < 5.0, "sampled frequencies follow the exact joint Born distribution");
}

} // namespace

int main(int argc, char **argv)
{
    const int n = argc > 1 ? std::stoi(argv[1]) : 12;
    const int timesteps = argc > 2 ? std::stoi(argv[2]) : 6;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 2024u;

    if (argc > 99)
    {
        cudaq::get_state(KeepLayerHelpersAlive{}, 2, std::vector<mipt::MmsLayer>{},
                         std::vector<mipt::HaarLayer>{}, std::vector<mipt::RppuLayer>{},
                         std::vector<mipt::RfgsLayer>{});
    }

    const bool fp64 = MIPT_CUDAQ_PRECISION == 64;
    std::cout << "cusv_equiv (CUDA-Q build precision fp" << MIPT_CUDAQ_PRECISION << ")\n";
    std::cout << "-- part 1: unitary evolution, p=0 --\n";

    try
    {
        for (int block : {2, 6})
        {
            for (const auto type : {mipt::CircuitType::MMS, mipt::CircuitType::Haar,
                                    mipt::CircuitType::FermionRPPU, mipt::CircuitType::QubitRPPU})
            {
                run_unitary_case(type, n, timesteps, seed, fp64, block);
            }
        }

        std::cout << "-- part 2: measurement layer --\n";
        run_measurement_case(n, seed, fp64);
        run_measurement_case(n, seed + 5u, fp64);
        run_sampling_statistics(10, seed, fp64);
    }
    catch (const std::exception &e)
    {
        std::cout << "  EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    if (failures == 0)
    {
        std::cout << "cusv_equiv: PASS\n";
        return 0;
    }
    std::cout << "cusv_equiv: " << failures << " FAILURE(S)\n";
    return 1;
}

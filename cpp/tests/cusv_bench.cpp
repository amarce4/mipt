// Times a realistic monitored trajectory through the CUDA-Q path and the
// cuStateVec engine. Both run the same circuit at the same p; only the engine
// differs.
//
// Usage: ./cusv_bench [n] [timesteps] [p] [circ_type]

#include "mipt/circuit.hpp"
#include "mipt/cusv/engine.hpp"
#include "mipt/probed.hpp"

#include <cudaq.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace
{

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

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char **argv)
{
    const int n = argc > 1 ? std::stoi(argv[1]) : 18;
    const int timesteps = argc > 2 ? std::stoi(argv[2]) : 16;
    const double p = argc > 3 ? std::stod(argv[3]) : 0.15;
    const int circ = argc > 4 ? std::stoi(argv[4]) : 2;

    if (argc > 99)
    {
        cudaq::get_state(KeepLayerHelpersAlive{}, 2, std::vector<mipt::MmsLayer>{},
                         std::vector<mipt::HaarLayer>{}, std::vector<mipt::RppuLayer>{},
                         std::vector<mipt::RfgsLayer>{});
    }

    const bool fp64 = MIPT_CUDAQ_PRECISION == 64;
    const auto type = mipt::parse_circuit_type(circ);
    std::printf("cusv_bench n=%d timesteps=%d p=%.3f circuit=%s precision=fp%d\n", n, timesteps, p,
                mipt::circuit_type_name(type).data(), MIPT_CUDAQ_PRECISION);

    // Build one shared layer history so both engines run the same circuit.
    std::mt19937 rng(20260728u);
    std::vector<mipt::RppuLayer> rppu_layers(static_cast<std::size_t>(timesteps));
    std::vector<mipt::HaarLayer> haar_layers(static_cast<std::size_t>(timesteps));
    const bool is_rppu = type == mipt::CircuitType::FermionRPPU || type == mipt::CircuitType::QubitRPPU;
    const mipt::RppuBoundaryMode mode = type == mipt::CircuitType::QubitRPPU
                                            ? mipt::RppuBoundaryMode::QubitDirect
                                            : mipt::RppuBoundaryMode::FermionicJWString;
    for (int s = 0; s < timesteps; ++s)
    {
        if (is_rppu)
        {
            mipt::fill_rppu_layer(rppu_layers[static_cast<std::size_t>(s)], n, (s % 2) != 0, p, true,
                                  rng, mode);
        }
        else
        {
            mipt::fill_haar_layer(haar_layers[static_cast<std::size_t>(s)], n, s % 2, p, true, rng);
        }
    }

    // --- CUDA-Q path: one get_state per timestep, exactly as advance() does.
    {
        auto t0 = std::chrono::steady_clock::now();
        auto state = is_rppu ? cudaq::get_state(mipt::RppuKernel1D{}, n,
                                                std::vector<mipt::RppuLayer>{rppu_layers[0]})
                             : cudaq::get_state(mipt::HaarKernel1D{}, n,
                                                std::vector<mipt::HaarLayer>{haar_layers[0]});
        for (int s = 1; s < timesteps; ++s)
        {
            if (is_rppu)
            {
                std::vector<mipt::RppuLayer> one{rppu_layers[static_cast<std::size_t>(s)]};
                state = cudaq::get_state(mipt::probed::RppuAdvanceKernel{}, &state, n, one);
            }
            else
            {
                std::vector<mipt::HaarLayer> one{haar_layers[static_cast<std::size_t>(s)]};
                state = cudaq::get_state(mipt::probed::HaarAdvanceKernel{}, &state, n, one);
            }
        }
        const double elapsed = seconds_since(t0);
        std::printf("  CUDA-Q  (gate-by-gate, per-timestep state copy) : %8.3f s  (%.1f ms/layer)\n",
                    elapsed, 1000.0 * elapsed / timesteps);
    }

    // --- cuStateVec engine: persistent buffer, fused bonds, batched measure.
    for (int block : {2, 4, 6, 8})
    {
        mipt::cusv::Engine engine(fp64);
        engine.reset(n);
        std::mt19937 draw_rng(4242u);
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < timesteps; ++s)
        {
            const auto ops = is_rppu
                                 ? mipt::cusv::build_rppu_layer_ops(
                                       rppu_layers[static_cast<std::size_t>(s)], n)
                                 : mipt::cusv::build_haar_layer_ops(
                                       haar_layers[static_cast<std::size_t>(s)], n);
            engine.apply_ops(ops.ops, block);
            engine.measure_layer(ops.measure_sites, draw_rng);
        }
        engine.synchronize();
        const double elapsed = seconds_since(t0);
        std::printf("  cusv    (block=%d)                              : %8.3f s  (%.1f ms/layer)\n",
                    block, elapsed, 1000.0 * elapsed / timesteps);
    }

    return 0;
}

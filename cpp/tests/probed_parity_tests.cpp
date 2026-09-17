// Validation for the probe-side parity-sector encoding
// (probed::CircuitWorkspace1D::advance_parity_sector).
//
// Before any reference is attached, a probe register is the bare system
// tensored with idle |0> ancillas, so a parity-preserving circuit keeps it in
// the even global-parity sector and the prefix can be simulated on 2^(N-1)
// amplitudes inside the caller's 2^(N+probes) buffer. The rewriting itself is
// pinned against a full-state host reference by cusv_parity_tests; what is new
// here is the embedding -- running half a system's worth of amplitudes in the
// low corner of a wider register, expanding into it, and refusing when the
// premise does not hold.
//
//   Part 1  p = 0: the encoded prefix reproduces the full-width path
//           amplitude by amplitude. Deterministic, so this is an equality.
//   Part 2  Structure: all of the weight sits in [0, 2^N) (the ancillas are
//           untouched) and every odd-parity system amplitude is exactly zero.
//   Part 3  Measured prefixes: normalization, the same two structural
//           invariants, and agreement of the sampled entropy distribution
//           with the full-width path.
//   Part 4  Refusals, each of which must leave the buffer untouched: a state
//           that is not |0...0>, a register with no room for an ancilla, a
//           circuit that is not parity preserving, and MIPT_CUSV_PARITY=0.
//
// Usage: ./probed_parity_tests [n] [timesteps] [seed]

#include "mipt/probed.hpp"

#include <cudaq.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

// probed/kernels.hpp is deliberately not included: it drags in the MOSEK shim
// and the ancilla reduction kernels, and all this needs from it is the product
// state its InitializeProductKernel builds.
struct ProductKernel
{
    void operator()(int total_qubits) __qpu__ { cudaq::qvector q(total_qubits); }
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

std::vector<std::complex<double>> to_host(cudaq::state &state)
{
    const auto tensor = state.get_tensor();
    const auto elements = tensor.get_num_elements();
    std::vector<std::complex<double>> out(elements);
    if (state.get_precision() == cudaq::SimulationState::precision::fp64)
    {
        state.to_host(out.data(), elements);
        return out;
    }
    std::vector<std::complex<float>> narrow(elements);
    state.to_host(narrow.data(), elements);
    for (std::size_t i = 0; i < elements; ++i)
    {
        out[i] = std::complex<double>(narrow[i].real(), narrow[i].imag());
    }
    return out;
}

cudaq::state product_state(int total_qubits)
{
    return cudaq::get_state(ProductKernel{}, total_qubits);
}

double max_difference(const std::vector<std::complex<double>> &a,
                      const std::vector<std::complex<double>> &b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

// Weight outside the system block, i.e. in any basis state where an ancilla
// carries an excitation. The prefix never touches them, so this is exactly 0.
double ancilla_weight(const std::vector<std::complex<double>> &psi, int n)
{
    double total = 0.0;
    for (std::size_t i = std::size_t{1} << n; i < psi.size(); ++i)
    {
        total += std::norm(psi[i]);
    }
    return total;
}

// Weight in the odd global-parity sector of the system block. This is the
// premise the encoding rests on, so it is checked rather than assumed.
double odd_sector_weight(const std::vector<std::complex<double>> &psi, int n)
{
    double total = 0.0;
    const std::size_t block = std::size_t{1} << n;
    for (std::size_t i = 0; i < block && i < psi.size(); ++i)
    {
        if ((__builtin_popcountll(i) & 1) != 0)
        {
            total += std::norm(psi[i]);
        }
    }
    return total;
}

// Probability that `site` is occupied, summed over the system block only.
double occupation(const std::vector<std::complex<double>> &psi, int n, int site)
{
    double occupied = 0.0;
    for (std::size_t i = 0; i < (std::size_t{1} << n); ++i)
    {
        if (((i >> site) & 1u) != 0u)
        {
            occupied += std::norm(psi[i]);
        }
    }
    return occupied;
}

double norm_of(const std::vector<std::complex<double>> &psi)
{
    double total = 0.0;
    for (const auto &value : psi)
    {
        total += std::norm(value);
    }
    return total;
}

} // namespace

int main(int argc, char *argv[])
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 10;
    const int timesteps = argc > 2 ? std::atoi(argv[2]) : 8;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 20260917u;
    const int probes = 2;
    const int total = n + probes;

    std::cout << "probed parity-sector prefix: n=" << n << " timesteps=" << timesteps
              << " probes=" << probes << " seed=" << seed << "\n";

    constexpr bool fp64 = MIPT_CUDAQ_PRECISION == 64;
    const double tol = fp64 ? 1e-13 : 2e-5;

    // MIPT_CUSV_PARITY=0 is the A/B control: every "is it taken" assertion
    // below flips with it, so the same binary pins both arms.
    const char *switch_value = std::getenv("MIPT_CUSV_PARITY");
    const bool expect_encoded = switch_value == nullptr || std::string(switch_value) != "0";
    std::cout << "  MIPT_CUSV_PARITY expects the encoded prefix to be "
              << (expect_encoded ? "taken" : "refused") << "\n";

    // ---------------------------------------------------------- Parts 1 & 2
    {
        mipt::probed::CircuitWorkspace1D workspace;
        workspace.seed(seed);
        workspace.prepare(n, timesteps, 0.0, mipt::CircuitType::FermionRPPU);

        auto encoded_state = product_state(total);
        const auto before = to_host(encoded_state);
        const bool took_it = workspace.advance_parity_sector(encoded_state, timesteps);
        check(took_it == expect_encoded, "MIPT_CUSV_PARITY decides whether the prefix is encoded");
        const auto encoded = to_host(encoded_state);

        if (!took_it)
        {
            check(max_difference(encoded, before) == 0.0,
                  "a refused prefix leaves the buffer untouched");
        }
        else
        {
            auto plain_state = product_state(total);
            workspace.advance(plain_state, 0, timesteps);
            const auto plain = to_host(plain_state);

            const double diff = max_difference(encoded, plain);
            std::cout << "  p=0 max |encoded - full| = " << diff << "\n";
            check(diff <= tol, "the encoded prefix reproduces the full-width path at p=0");
            check(ancilla_weight(encoded, n) == 0.0, "the ancillas stay exactly |0>");
            check(odd_sector_weight(encoded, n) == 0.0,
                  "the odd global-parity sector is exactly empty");
            check(std::abs(norm_of(encoded) - 1.0) <= tol, "the encoded prefix stays normalized");
        }
    }

    // ---------------------------------------------------------------- Part 3
    if (expect_encoded)
    {
        // The two paths draw their outcomes from differently chunked joint
        // distributions, so they are different trajectories and only the
        // ensemble is comparable -- the same situation every backend A/B in
        // this tree is in. Mid-chain occupation is compared as a z-score over
        // its own sample spread rather than against a hand-picked tolerance.
        const int trajectories = 200;
        double encoded_sum = 0.0, encoded_sq = 0.0;
        double plain_sum = 0.0, plain_sq = 0.0;
        for (int trajectory = 0; trajectory < trajectories; ++trajectory)
        {
            mipt::probed::CircuitWorkspace1D workspace;
            workspace.seed(seed + static_cast<unsigned>(trajectory));
            workspace.prepare(n, timesteps, 0.2, mipt::CircuitType::FermionRPPU);

            auto encoded_state = product_state(total);
            const bool took_it = workspace.advance_parity_sector(encoded_state, timesteps);
            check(took_it, "the encoded prefix is taken on a measured RPPU trajectory");
            const auto encoded = to_host(encoded_state);
            check(std::abs(norm_of(encoded) - 1.0) <= tol,
                  "a measured encoded prefix stays normalized");
            check(ancilla_weight(encoded, n) == 0.0,
                  "a measured encoded prefix leaves the ancillas at |0>");
            check(odd_sector_weight(encoded, n) == 0.0,
                  "a measured encoded prefix stays in the even sector");

            auto plain_state = product_state(total);
            workspace.advance(plain_state, 0, timesteps);
            const auto plain = to_host(plain_state);
            check(std::abs(norm_of(plain) - 1.0) <= tol, "the full-width path stays normalized");

            const double a = occupation(encoded, n, n / 2);
            const double b = occupation(plain, n, n / 2);
            encoded_sum += a;
            encoded_sq += a * a;
            plain_sum += b;
            plain_sq += b * b;
            if (failures > 8)
            {
                break;
            }
        }
        const double count = trajectories;
        const double encoded_mean = encoded_sum / count;
        const double plain_mean = plain_sum / count;
        const double encoded_var = (encoded_sq / count - encoded_mean * encoded_mean) / (count - 1.0);
        const double plain_var = (plain_sq / count - plain_mean * plain_mean) / (count - 1.0);
        const double sigma = std::sqrt(std::max(encoded_var + plain_var, 1e-300));
        const double z = (encoded_mean - plain_mean) / sigma;
        std::cout << "  <n_mid> encoded=" << encoded_mean << " full=" << plain_mean
                  << " z=" << z << " over " << trajectories << " trajectories\n";
        check(std::abs(z) < 4.0, "the two paths agree on the mean mid-chain occupation");
    }

    // ---------------------------------------------------------------- Part 4
    {
        mipt::probed::CircuitWorkspace1D workspace;
        workspace.seed(seed);
        workspace.prepare(n, timesteps, 0.1, mipt::CircuitType::FermionRPPU);

        // A register with no ancilla is the reference-encoded probe state's
        // shape; it must not be mistaken for an idle-ancilla prefix.
        auto narrow_state = product_state(n);
        const auto narrow_before = to_host(narrow_state);
        check(!workspace.advance_parity_sector(narrow_state, timesteps),
              "a register with no ancilla is refused");
        check(max_difference(to_host(narrow_state), narrow_before) == 0.0,
              "the refused narrow register is left untouched");

        // Not |0...0>: the probe-encoded state carries weight in both sectors
        // and is indistinguishable from the outside, so the amplitude check is
        // the only thing standing between it and a wrong answer.
        auto superposed = cudaq::get_state(mipt::probed::ParityProbeInitKernel{}, total, 0);
        const auto superposed_before = to_host(superposed);
        check(!workspace.advance_parity_sector(superposed, timesteps),
              "a state that is not |0...0> is refused");
        check(max_difference(to_host(superposed), superposed_before) == 0.0,
              "the refused superposed state is left untouched");
    }

    {
        mipt::probed::CircuitWorkspace1D workspace;
        workspace.seed(seed);
        workspace.prepare(n, timesteps, 0.1, mipt::CircuitType::Haar);
        auto state = product_state(total);
        const auto before = to_host(state);
        check(!workspace.advance_parity_sector(state, timesteps),
              "a circuit that does not preserve parity is refused");
        check(max_difference(to_host(state), before) == 0.0,
              "the refused Haar register is left untouched");
    }

    if (failures == 0)
    {
        std::cout << "probed parity-sector prefix: PASS\n";
        return 0;
    }
    std::cout << "probed parity-sector prefix: " << failures << " failure(s)\n";
    return 1;
}

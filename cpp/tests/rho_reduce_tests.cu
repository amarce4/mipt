// Correctness tests for the fused two- and three-site reduction sweeps in
// src/cuda/rho2_reduce.cu and src/cuda/rho_reduce.cu.
//
// The reference here is deliberately *not* a transcription of the kernels'
// sign machinery. The kernels split the Jordan-Wigner sign into a
// "kept modes reordered among themselves" parity and a per-environment-mode
// crossing mask; this file instead builds the sign by counting inversions of
// the permutation that takes the ascending list of occupied modes to
//
//     (occupied kept modes, in the retained order) ++ (occupied env modes, ascending)
//
// which is the convention rho_reduce.cu documents, expressed from first
// principles. The two agree only if the mask decomposition is right.
//
// Trajectories are not reproducible (see CLAUDE.md), so a reduction change can
// only be validated at this level: build a state, call the entry points on it,
// compare against a reference computed on the same amplitudes.
//
// Covered: both traces, both precisions, all three modes (ordinary, fermionic,
// and the fused dual-trace path), contiguous / distant / wrapped / **unsorted**
// retained sets -- mode 5 passes descending sets on purpose, and only the
// environment embedding may reorder them.

#include "mipt/cuda/rho2_reduce.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/cuda/rho_reduce.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
using Complex = std::complex<double>;

int failures = 0;

struct Rng
{
    std::uint64_t state;
    double uniform()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    }
    double normal()
    {
        const double u1 = std::fmax(1.0e-12, uniform());
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// A dense normalized state whose amplitudes are exactly representable in fp32,
// so that the fp32 kernels and the fp64 reference see identical inputs and any
// disagreement is accumulation, not rounding of the state.
std::vector<Complex> random_state(int n, std::uint64_t seed)
{
    Rng rng{seed};
    const std::size_t dim = std::size_t{1} << n;
    std::vector<Complex> psi(dim);
    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        const float re = static_cast<float>(rng.normal());
        const float im = static_cast<float>(rng.normal());
        psi[i] = Complex(re, im);
        norm += static_cast<double>(re) * re + static_cast<double>(im) * im;
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::size_t i = 0; i < dim; ++i)
    {
        psi[i] = Complex(static_cast<float>(psi[i].real() * scale),
                         static_cast<float>(psi[i].imag() * scale));
    }
    return psi;
}

// Parity of the permutation taking the ascending occupied-mode list of `index`
// to (occupied kept modes in retained order) ++ (occupied env modes ascending).
bool jordan_wigner_odd(std::uint64_t index, const std::vector<int> &kept, int n)
{
    std::vector<int> ascending;
    for (int q = 0; q < n; ++q)
    {
        if ((index >> q) & 1u)
        {
            ascending.push_back(q);
        }
    }

    std::vector<int> target;
    for (int q : kept)
    {
        if ((index >> q) & 1u)
        {
            target.push_back(q);
        }
    }
    for (int q = 0; q < n; ++q)
    {
        if (std::find(kept.begin(), kept.end(), q) != kept.end())
        {
            continue;
        }
        if ((index >> q) & 1u)
        {
            target.push_back(q);
        }
    }

    // Rewrite the target as positions into the ascending list, then count
    // inversions: that count's parity is the permutation's sign.
    std::vector<int> order;
    order.reserve(target.size());
    for (int q : target)
    {
        order.push_back(
            static_cast<int>(std::find(ascending.begin(), ascending.end(), q) - ascending.begin()));
    }
    int inversions = 0;
    for (std::size_t i = 0; i < order.size(); ++i)
    {
        for (std::size_t j = i + 1; j < order.size(); ++j)
        {
            if (order[i] > order[j])
            {
                ++inversions;
            }
        }
    }
    return (inversions & 1) != 0;
}

std::uint64_t embed_retained(int basis, const std::vector<int> &kept)
{
    std::uint64_t offset = 0;
    for (std::size_t k = 0; k < kept.size(); ++k)
    {
        if ((basis >> k) & 1)
        {
            offset |= std::uint64_t{1} << kept[k];
        }
    }
    return offset;
}

std::vector<Complex> reference_rdm(const std::vector<Complex> &psi, int n,
                                   const std::vector<int> &kept, bool fermionic)
{
    const int dim = 1 << static_cast<int>(kept.size());
    std::vector<Complex> rho(static_cast<std::size_t>(dim) * dim, Complex(0.0, 0.0));

    std::uint64_t kept_mask = 0;
    for (int q : kept)
    {
        kept_mask |= std::uint64_t{1} << q;
    }

    std::vector<Complex> column(dim);
    const std::uint64_t full = std::uint64_t{1} << n;
    for (std::uint64_t env = 0; env < full; ++env)
    {
        if ((env & kept_mask) != 0)
        {
            continue; // not an environment configuration
        }
        for (int r = 0; r < dim; ++r)
        {
            const std::uint64_t index = env | embed_retained(r, kept);
            const double sign = (fermionic && jordan_wigner_odd(index, kept, n)) ? -1.0 : 1.0;
            column[r] = sign * psi[index];
        }
        for (int r = 0; r < dim; ++r)
        {
            for (int c = 0; c < dim; ++c)
            {
                rho[static_cast<std::size_t>(r) * dim + c] += column[r] * std::conj(column[c]);
            }
        }
    }
    return rho;
}

double max_deviation(const std::vector<double> &got, const std::vector<Complex> &want)
{
    double worst = 0.0;
    for (std::size_t e = 0; e < want.size(); ++e)
    {
        worst = std::fmax(worst, std::fabs(got[2 * e + 0] - want[e].real()));
        worst = std::fmax(worst, std::fabs(got[2 * e + 1] - want[e].imag()));
    }
    return worst;
}

double max_difference(const std::vector<double> &a, const std::vector<double> &b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

std::string describe(const std::vector<int> &modes)
{
    std::string text = "{";
    for (std::size_t i = 0; i < modes.size(); ++i)
    {
        text += (i ? "," : "") + std::to_string(modes[i]);
    }
    return text + "}";
}

void expect(bool condition, const char *label, const std::vector<int> &modes, bool fp64,
            double value, double tolerance)
{
    if (!condition)
    {
        ++failures;
    }
    std::printf("  %s %-34s %-14s %s  %.3e (tol %.1e)\n", condition ? "ok  " : "FAIL", label,
                describe(modes).c_str(), fp64 ? "f64" : "f32", value, tolerance);
}

// Device state, uploaded once per precision.
template <typename Real>
struct DeviceState
{
    void *data = nullptr;
    explicit DeviceState(const std::vector<Complex> &psi)
    {
        std::vector<Real> host(2 * psi.size());
        for (std::size_t i = 0; i < psi.size(); ++i)
        {
            host[2 * i + 0] = static_cast<Real>(psi[i].real());
            host[2 * i + 1] = static_cast<Real>(psi[i].imag());
        }
        cudaMalloc(&data, host.size() * sizeof(Real));
        cudaMemcpy(data, host.data(), host.size() * sizeof(Real), cudaMemcpyHostToDevice);
    }
    ~DeviceState() { cudaFree(data); }
};

void check_pair(const std::vector<Complex> &psi, int n, const std::vector<int> &kept, bool fp64,
                double tolerance)
{
    const std::size_t values = 32; // 2 * 4 * 4
    std::vector<double> ordinary(values), fermionic(values);
    std::vector<double> both_ordinary(values), both_fermionic(values);
    int status = 0;

    const auto run = [&](auto &state) {
        if (fp64)
        {
            status |= mipt_cuda_rho2_subsystems_complex_f64(state.data, n, kept.data(), 1,
                                                            ordinary.data());
            status |= mipt_cuda_rho2_subsystems_complex_fermion_f64(state.data, n, kept.data(), 1,
                                                                    fermionic.data());
            status |= mipt_cuda_rho2_subsystems_complex_both_f64(
                state.data, n, kept.data(), 1, both_ordinary.data(), both_fermionic.data());
        }
        else
        {
            status |= mipt_cuda_rho2_subsystems_complex_f32(state.data, n, kept.data(), 1,
                                                            ordinary.data());
            status |= mipt_cuda_rho2_subsystems_complex_fermion_f32(state.data, n, kept.data(), 1,
                                                                    fermionic.data());
            status |= mipt_cuda_rho2_subsystems_complex_both_f32(
                state.data, n, kept.data(), 1, both_ordinary.data(), both_fermionic.data());
        }
    };
    if (fp64)
    {
        DeviceState<double> state(psi);
        run(state);
    }
    else
    {
        DeviceState<float> state(psi);
        run(state);
    }
    if (status != 0)
    {
        ++failures;
        std::printf("  FAIL rho2 kernel returned status=%d for %s\n", status,
                    describe(kept).c_str());
        return;
    }

    expect(max_deviation(ordinary, reference_rdm(psi, n, kept, false)) <= tolerance,
           "rho2 ordinary vs reference", kept, fp64,
           max_deviation(ordinary, reference_rdm(psi, n, kept, false)), tolerance);
    expect(max_deviation(fermionic, reference_rdm(psi, n, kept, true)) <= tolerance,
           "rho2 fermionic vs reference", kept, fp64,
           max_deviation(fermionic, reference_rdm(psi, n, kept, true)), tolerance);
    expect(max_difference(both_ordinary, ordinary) <= tolerance, "rho2 both vs single (ordinary)",
           kept, fp64, max_difference(both_ordinary, ordinary), tolerance);
    expect(max_difference(both_fermionic, fermionic) <= tolerance,
           "rho2 both vs single (fermionic)", kept, fp64,
           max_difference(both_fermionic, fermionic), tolerance);
}

void check_triple(const std::vector<Complex> &psi, int n, const std::vector<int> &kept, bool fp64,
                  double tolerance)
{
    const std::size_t values = 128; // 2 * 8 * 8
    std::vector<double> ordinary(values), fermionic(values);
    std::vector<double> both_ordinary(values), both_fermionic(values);
    int status = 0;

    const auto run = [&](auto &state) {
        if (fp64)
        {
            status |= mipt_cuda_rho3_subsystems_complex_f64(state.data, n, kept.data(), 1,
                                                            ordinary.data());
            status |= mipt_cuda_rho3_subsystems_complex_fermion_f64(state.data, n, kept.data(), 1,
                                                                    fermionic.data());
            status |= mipt_cuda_rho3_subsystems_complex_both_f64(
                state.data, n, kept.data(), 1, both_ordinary.data(), both_fermionic.data());
        }
        else
        {
            status |= mipt_cuda_rho3_subsystems_complex_f32(state.data, n, kept.data(), 1,
                                                            ordinary.data());
            status |= mipt_cuda_rho3_subsystems_complex_fermion_f32(state.data, n, kept.data(), 1,
                                                                    fermionic.data());
            status |= mipt_cuda_rho3_subsystems_complex_both_f32(
                state.data, n, kept.data(), 1, both_ordinary.data(), both_fermionic.data());
        }
    };
    if (fp64)
    {
        DeviceState<double> state(psi);
        run(state);
    }
    else
    {
        DeviceState<float> state(psi);
        run(state);
    }
    if (status != 0)
    {
        ++failures;
        std::printf("  FAIL rho3 kernel returned status=%d for %s\n", status,
                    describe(kept).c_str());
        return;
    }

    expect(max_deviation(ordinary, reference_rdm(psi, n, kept, false)) <= tolerance,
           "rho3 ordinary vs reference", kept, fp64,
           max_deviation(ordinary, reference_rdm(psi, n, kept, false)), tolerance);
    expect(max_deviation(fermionic, reference_rdm(psi, n, kept, true)) <= tolerance,
           "rho3 fermionic vs reference", kept, fp64,
           max_deviation(fermionic, reference_rdm(psi, n, kept, true)), tolerance);
    expect(max_difference(both_ordinary, ordinary) <= tolerance, "rho3 both vs single (ordinary)",
           kept, fp64, max_difference(both_ordinary, ordinary), tolerance);
    expect(max_difference(both_fermionic, fermionic) <= tolerance,
           "rho3 both vs single (fermionic)", kept, fp64,
           max_difference(both_fermionic, fermionic), tolerance);
}

// The fermionic sign only bites when an environment mode sits between two
// retained ones, so a test using only contiguous blocks would pass with the
// signs dropped entirely. This asserts they are live.
int occupied_below(std::uint64_t index, int mode)
{
    return __builtin_popcountll(index & ((std::uint64_t{1} << mode) - 1u));
}

// <psi| c_a^dag c_b |psi>, straight from the anticommutation algebra: there is
// no reduced density matrix anywhere in this, which is the point.
//   c_b |x> = (-1)^{occupied below b} |x with bit b cleared>   when x_b = 1
//   c_a^dag |y> = (-1)^{occupied below a} |y with bit a set>   when y_a = 0
Complex hopping_correlator(const std::vector<Complex> &psi, int a, int b)
{
    Complex total(0.0, 0.0);
    for (std::uint64_t x = 0; x < psi.size(); ++x)
    {
        if (((x >> b) & 1u) == 0u) { continue; }
        int sign = (occupied_below(x, b) & 1) ? -1 : 1;
        const std::uint64_t y = x & ~(std::uint64_t{1} << b);
        if (((y >> a) & 1u) != 0u) { continue; }
        sign *= (occupied_below(y, a) & 1) ? -1 : 1;
        total += std::conj(psi[y | (std::uint64_t{1} << a)]) * psi[x] * static_cast<double>(sign);
    }
    return total;
}

// <psi| c_a c_b |psi>.
Complex pairing_correlator(const std::vector<Complex> &psi, int a, int b)
{
    Complex total(0.0, 0.0);
    for (std::uint64_t x = 0; x < psi.size(); ++x)
    {
        if (((x >> b) & 1u) == 0u) { continue; }
        int sign = (occupied_below(x, b) & 1) ? -1 : 1;
        const std::uint64_t y = x & ~(std::uint64_t{1} << b);
        if (((y >> a) & 1u) == 0u) { continue; }
        sign *= (occupied_below(y, a) & 1) ? -1 : 1;
        total += std::conj(psi[y & ~(std::uint64_t{1} << a)]) * psi[x] * static_cast<double>(sign);
    }
    return total;
}

mipt::dist::Matrix4 as_matrix4(const std::vector<Complex> &rho)
{
    mipt::dist::Matrix4 m{};
    for (std::size_t e = 0; e < 16; ++e) { m[e] = rho[e]; }
    return m;
}

// Closes the chain behind the g2/f2 columns of the k=2 CSV.
//
// check_pair pins the production kernels against reference_rdm; this pins
// dist::two_point_correlators(reference_rdm) against the operator algebra
// above. Together they say that |<c_i^dag c_j>|^2 and |<c_i c_j>|^2 read off
// two entries of the *fermionic* pair RDM are the real thing.
//
// It also checks that the ordinary-trace RDM gives something else, which is
// the whole reason the fermionic trace exists here: the Jordan-Wigner string
// running between the two modes is what it restores.
void check_two_point_correlators(const std::vector<Complex> &psi, int n)
{
    const std::vector<std::vector<int>> pairs{{0, 1}, {0, 5}, {2, n - 1}, {4, 7}};
    for (const auto &kept : pairs)
    {
        const auto correlators = mipt::dist::two_point_correlators(
            as_matrix4(reference_rdm(psi, n, kept, true)));
        const double g2 = std::norm(hopping_correlator(psi, kept[0], kept[1]));
        const double f2 = std::norm(pairing_correlator(psi, kept[0], kept[1]));
        const double worst = std::fmax(std::fabs(correlators.g2 - g2),
                                       std::fabs(correlators.f2 - f2));
        if (!(worst < 1.0e-12))
        {
            ++failures;
            std::printf("  FAIL two-point correlators (%d,%d): rdm {%.9f, %.9f} vs operators "
                        "{%.9f, %.9f}\n", kept[0], kept[1], correlators.g2, correlators.f2, g2, f2);
        }
        else
        {
            std::printf("  ok   |G|^2=%.6f |F|^2=%.6f from the fermionic pair RDM matches the "
                        "operator algebra for (%d,%d): max|d|=%.2e\n",
                        g2, f2, kept[0], kept[1], worst);
        }

        const auto qubit = mipt::dist::two_point_correlators(
            as_matrix4(reference_rdm(psi, n, kept, false)));
        const bool distant = std::abs(kept[1] - kept[0]) > 1;
        const double gap = std::fmax(std::fabs(qubit.g2 - correlators.g2),
                                     std::fabs(qubit.f2 - correlators.f2));
        if (distant && !(gap > 1.0e-6))
        {
            ++failures;
            std::printf("  FAIL ordinary-trace correlators match the fermionic ones at (%d,%d); "
                        "the Jordan-Wigner string is not being applied\n", kept[0], kept[1]);
        }
    }
}

void check_fermionic_trace_is_live(const std::vector<Complex> &psi, int n)
{
    const std::vector<int> pair{1, 6};
    const std::vector<int> triple{1, 5, 9};

    const auto qubit_pair = reference_rdm(psi, n, pair, false);
    const auto fermi_pair = reference_rdm(psi, n, pair, true);
    double worst = 0.0;
    for (std::size_t e = 0; e < qubit_pair.size(); ++e)
    {
        worst = std::fmax(worst, std::abs(qubit_pair[e] - fermi_pair[e]));
    }
    if (worst < 1.0e-3)
    {
        ++failures;
        std::printf("  FAIL fermionic pair trace is indistinguishable from the qubit one\n");
    }
    else
    {
        std::printf("  ok   fermionic trace changes a scattered pair RDM: max|d|=%.3e\n", worst);
    }

    const auto qubit_triple = reference_rdm(psi, n, triple, false);
    const auto fermi_triple = reference_rdm(psi, n, triple, true);
    worst = 0.0;
    for (std::size_t e = 0; e < qubit_triple.size(); ++e)
    {
        worst = std::fmax(worst, std::abs(qubit_triple[e] - fermi_triple[e]));
    }
    if (worst < 1.0e-3)
    {
        ++failures;
        std::printf("  FAIL fermionic triple trace is indistinguishable from the qubit one\n");
    }
    else
    {
        std::printf("  ok   fermionic trace changes a scattered triple RDM: max|d|=%.3e\n", worst);
    }
}
} // namespace

int main()
{
    const int n = 11;
    const auto psi = random_state(n, 0x5eed1234u);

    // Adjacent, distant, wrapped, and two deliberately unsorted retained sets.
    const std::vector<std::vector<int>> pairs{
        {0, 1}, {0, 5}, {5, 0}, {n - 1, 0}, {3, n - 2}, {n - 2, 3}};
    const std::vector<std::vector<int>> triples{
        {0, 1, 2}, {0, 4, 8}, {8, 4, 0}, {n - 1, 0, 1}, {2, 7, 5}, {9, 2, 6}};

    std::printf("rho2/rho3 fused reduction sweeps, N=%d\n", n);
    check_fermionic_trace_is_live(psi, n);
    check_two_point_correlators(psi, n);

    for (bool fp64 : {false, true})
    {
        // The fp32 sweep accumulates in fp32 and promotes periodically; the
        // fp64 sweep accumulates in fp64 throughout. See reduce_sweep.cuh.
        const double tolerance = fp64 ? 1.0e-12 : 1.0e-7;
        for (const auto &kept : pairs)
        {
            check_pair(psi, n, kept, fp64, tolerance);
        }
        for (const auto &kept : triples)
        {
            check_triple(psi, n, kept, fp64, tolerance);
        }
    }

    // Many subsystems in one call: the batched path indexes partials by
    // subsystem, and a wrong stride there would not show up above.
    {
        std::vector<int> flat_pairs, flat_triples;
        for (const auto &kept : pairs)
        {
            flat_pairs.insert(flat_pairs.end(), kept.begin(), kept.end());
        }
        for (const auto &kept : triples)
        {
            flat_triples.insert(flat_triples.end(), kept.begin(), kept.end());
        }
        DeviceState<float> state(psi);
        std::vector<double> batch2(32 * pairs.size()), batch2f(32 * pairs.size());
        std::vector<double> batch3(128 * triples.size()), batch3f(128 * triples.size());
        int status = mipt_cuda_rho2_subsystems_complex_both_f32(
            state.data, n, flat_pairs.data(), static_cast<int>(pairs.size()), batch2.data(),
            batch2f.data());
        status |= mipt_cuda_rho3_subsystems_complex_both_f32(
            state.data, n, flat_triples.data(), static_cast<int>(triples.size()), batch3.data(),
            batch3f.data());
        if (status != 0)
        {
            ++failures;
            std::printf("  FAIL batched call returned status=%d\n", status);
        }
        double worst = 0.0;
        for (std::size_t s = 0; s < pairs.size(); ++s)
        {
            const auto want = reference_rdm(psi, n, pairs[s], true);
            std::vector<double> slice(batch2f.begin() + 32 * s, batch2f.begin() + 32 * (s + 1));
            worst = std::fmax(worst, max_deviation(slice, want));
        }
        for (std::size_t s = 0; s < triples.size(); ++s)
        {
            const auto want = reference_rdm(psi, n, triples[s], true);
            std::vector<double> slice(batch3f.begin() + 128 * s, batch3f.begin() + 128 * (s + 1));
            worst = std::fmax(worst, max_deviation(slice, want));
        }
        if (worst > 1.0e-7)
        {
            ++failures;
        }
        std::printf("  %s batched fermionic sweep, %zu pairs + %zu triples: %.3e\n",
                    worst <= 1.0e-7 ? "ok  " : "FAIL", pairs.size(), triples.size(), worst);
    }

    if (failures != 0)
    {
        std::printf("%d rho-reduce check(s) failed\n", failures);
        return 1;
    }
    std::printf("All rho-reduce sweep tests passed.\n");
    return 0;
}

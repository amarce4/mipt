// Equivalence tests for the gather + cuBLAS-herk reduction that replaced the
// per-output-element RDM kernels and the cusolverDnCgesvd half-cut entropy.
//
// Both replacements change *how* the reduction is accumulated, so they are
// checked against a host fp64 reference rather than against each other:
//
//   * mipt_cuda_rdm_subsystem_*  -- the old kernel accumulated in double from
//     an fp32 state; cuBLAS herk accumulates at the state's own precision.
//     This measures how much that costs.
//   * mipt_cuda_entropy_svd_*    -- forming the Gram matrix squares the
//     condition number, so the small end of the entanglement spectrum is the
//     thing at risk.  The reference entropy is computed from an fp64 RDM.
//
// Both the qubit and the fermionic trace are covered, at several cut sizes.

#include "mipt/cuda/tmi_cuda_rdm.hpp"
#include "mipt/cuda/tmi_cuda_svd.hpp"
#include "mipt/cuda/schmidt_gram.cuh"
#include "mipt/util/spectral.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// A generic entangled state: dense, no structure the reduction could exploit,
// and a broad Schmidt spectrum on every cut.
std::vector<Complex> random_state(int n, std::uint64_t seed)
{
    Rng rng{seed};
    const std::size_t dim = std::size_t{1} << n;
    std::vector<Complex> psi(dim);
    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        psi[i] = Complex(rng.normal(), rng.normal());
        norm += std::norm(psi[i]);
    }
    norm = std::sqrt(norm);
    for (auto &z : psi)
    {
        z /= norm;
    }
    return psi;
}

// Host reference: the same Schmidt gather the device does, reduced in double.
std::vector<Complex> reference_rdm(const std::vector<Complex> &psi,
                                   int n,
                                   const std::vector<int> &kept_modes,
                                   bool fermion_trace)
{
    const int kept_count = static_cast<int>(kept_modes.size());
    const std::size_t rows = std::size_t{1} << kept_count;
    const std::size_t cols = std::size_t{1} << (n - kept_count);

    int *env_modes = nullptr;
    mipt_gram::complement_modes_host(n, kept_modes.data(), kept_count, &env_modes);
    const int env_count = n - kept_count;

    std::uint64_t *row_offsets =
        mipt_gram::basis_offsets_host(kept_modes.data(), kept_count);
    std::uint64_t *env_offsets =
        mipt_gram::basis_offsets_host(env_modes, env_count);
    std::uint64_t *cross =
        fermion_trace ? mipt_gram::fermion_cross_masks_by_row_host(
                            kept_modes.data(), kept_count, env_modes, env_count, rows)
                      : static_cast<std::uint64_t *>(calloc(rows, sizeof(std::uint64_t)));

    std::vector<Complex> rho(rows * rows, Complex(0.0, 0.0));
    for (std::size_t col = 0; col < cols; ++col)
    {
        for (std::size_t r = 0; r < rows; ++r)
        {
            const double sr = mipt_gram::odd_popcount64(
                                  static_cast<std::uint64_t>(col) & cross[r]) ? -1.0 : 1.0;
            const Complex a = sr * psi[env_offsets[col] | row_offsets[r]];
            for (std::size_t c = 0; c < rows; ++c)
            {
                const double sc = mipt_gram::odd_popcount64(
                                      static_cast<std::uint64_t>(col) & cross[c]) ? -1.0 : 1.0;
                const Complex b = sc * psi[env_offsets[col] | row_offsets[c]];
                rho[r * rows + c] += a * std::conj(b);
            }
        }
    }

    free(env_modes);
    free(row_offsets);
    free(env_offsets);
    free(cross);
    return rho;
}

template <std::size_t N>
double reference_entropy_bits(const std::vector<Complex> &rho)
{
    std::array<Complex, N * N> flat{};
    for (std::size_t i = 0; i < N * N; ++i)
    {
        flat[i] = rho[i];
    }
    std::array<double, N> eigenvalues{};
    if (!mipt::util::hermitian_eigenvalues<N>(flat, eigenvalues))
    {
        std::fprintf(stderr, "FAIL reference eigensolver did not converge\n");
        ++failures;
        return 0.0;
    }
    double entropy = 0.0;
    for (const double lambda : eigenvalues)
    {
        if (lambda > 1.0e-15)
        {
            entropy -= lambda * std::log2(lambda);
        }
    }
    return entropy;
}

template <typename Real>
void *upload_state(const std::vector<Complex> &psi)
{
    std::vector<Real> flat(2 * psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i)
    {
        flat[2 * i] = static_cast<Real>(psi[i].real());
        flat[2 * i + 1] = static_cast<Real>(psi[i].imag());
    }
    void *device = nullptr;
    if (cudaMalloc(&device, flat.size() * sizeof(Real)) != cudaSuccess)
    {
        std::fprintf(stderr, "FAIL cudaMalloc state\n");
        std::exit(1);
    }
    cudaMemcpy(device, flat.data(), flat.size() * sizeof(Real), cudaMemcpyHostToDevice);
    return device;
}

void check_rdm(int n, const std::vector<int> &kept, bool fermionic, bool fp64,
               double tolerance)
{
    const auto psi = random_state(n, 0xABCDEF0123456789ULL + n + kept.size());
    const auto reference = reference_rdm(psi, n, kept, fermionic);
    const std::size_t rows = std::size_t{1} << kept.size();

    void *device = fp64 ? upload_state<double>(psi) : upload_state<float>(psi);
    std::vector<double> got(2 * rows * rows, 0.0);
    const int status =
        fp64 ? mipt_cuda_rdm_subsystem_f64(device, n, kept.data(),
                                           static_cast<int>(kept.size()),
                                           fermionic ? 1 : 0, got.data())
             : mipt_cuda_rdm_subsystem_f32(device, n, kept.data(),
                                           static_cast<int>(kept.size()),
                                           fermionic ? 1 : 0, got.data());
    cudaFree(device);

    if (status != 0)
    {
        std::fprintf(stderr, "FAIL rdm status=%d (n=%d kept=%zu %s %s)\n", status, n,
                     kept.size(), fermionic ? "fermi" : "qubit", fp64 ? "f64" : "f32");
        ++failures;
        return;
    }

    double worst = 0.0;
    double trace = 0.0;
    for (std::size_t i = 0; i < rows * rows; ++i)
    {
        worst = std::fmax(worst, std::abs(got[2 * i] - reference[i].real()));
        worst = std::fmax(worst, std::abs(got[2 * i + 1] - reference[i].imag()));
    }
    for (std::size_t i = 0; i < rows; ++i)
    {
        trace += got[2 * (i * rows + i)];
    }

    const char *verdict = (worst <= tolerance && std::abs(trace - 1.0) <= tolerance)
                              ? "ok  " : "FAIL";
    if (verdict[0] == 'F')
    {
        ++failures;
    }
    std::printf("  %s rdm  n=%2d kept=%zu %-5s %s  max|d|=%.3e  |tr-1|=%.3e\n",
                verdict, n, kept.size(), fermionic ? "fermi" : "qubit",
                fp64 ? "f64" : "f32", worst, std::abs(trace - 1.0));
}

template <std::size_t ROWS>
void check_entropy(int n, const std::vector<int> &kept, bool fermionic, bool fp64,
                   double tolerance)
{
    const auto psi = random_state(n, 0x1234567890ABCDEFULL + n + kept.size());
    const auto reference = reference_rdm(psi, n, kept, fermionic);
    const double expected = reference_entropy_bits<ROWS>(reference);

    void *device = fp64 ? upload_state<double>(psi) : upload_state<float>(psi);
    double got = 0.0;
    const int status =
        fp64 ? mipt_cuda_entropy_svd_f64(device, n, kept.data(),
                                         static_cast<int>(kept.size()),
                                         fermionic ? 1 : 0, &got)
             : mipt_cuda_entropy_svd_f32(device, n, kept.data(),
                                         static_cast<int>(kept.size()),
                                         fermionic ? 1 : 0, &got);
    cudaFree(device);

    if (status != 0)
    {
        std::fprintf(stderr, "FAIL entropy status=%d\n", status);
        ++failures;
        return;
    }
    const double delta = std::abs(got - expected);
    const char *verdict = (delta <= tolerance) ? "ok  " : "FAIL";
    if (verdict[0] == 'F')
    {
        ++failures;
    }
    std::printf("  %s ent  n=%2d kept=%zu %-5s %s  S=%.9f ref=%.9f  dS=%.3e\n",
                verdict, n, kept.size(), fermionic ? "fermi" : "qubit",
                fp64 ? "f64" : "f32", got, expected, delta);
}
} // namespace

// The fermionic Jordan-Wigner signs only bite when some environment mode sits
// below a retained one, so a retained set starting at mode 0 gives the same RDM
// under both traces.  Assert that a *scattered* set does not, which is what
// would catch fermion_trace being silently dropped on the way through the
// gather -- a failure the reference comparison cannot see, because the
// reference shares its cross-mask tables with the device path.
void check_fermionic_trace_is_live(int n, const std::vector<int> &kept)
{
    const auto psi = random_state(n, 0x5555AAAA5555AAAAULL);
    const std::size_t rows = std::size_t{1} << kept.size();
    void *device = upload_state<double>(psi);

    std::vector<double> qubit(2 * rows * rows, 0.0);
    std::vector<double> fermi(2 * rows * rows, 0.0);
    mipt_cuda_rdm_subsystem_f64(device, n, kept.data(),
                                static_cast<int>(kept.size()), 0, qubit.data());
    mipt_cuda_rdm_subsystem_f64(device, n, kept.data(),
                                static_cast<int>(kept.size()), 1, fermi.data());
    cudaFree(device);

    double spread = 0.0;
    for (std::size_t i = 0; i < qubit.size(); ++i)
    {
        spread = std::fmax(spread, std::abs(qubit[i] - fermi[i]));
    }
    const bool live = spread > 1.0e-6;
    if (!live)
    {
        ++failures;
    }
    std::printf("  %s fermionic trace changes a scattered RDM: max|d|=%.3e\n",
                live ? "ok  " : "FAIL", spread);
}

int main()
{
    std::printf("RDM export (herk accumulates at state precision):\n");
    for (bool fermionic : {false, true})
    {
        // Contiguous and scattered retained sets, since TMI uses both
        // (quarters are contiguous, AC is not).
        check_rdm(14, {0, 1, 2}, fermionic, false, 2.0e-6);
        check_rdm(14, {3, 7, 11}, fermionic, false, 2.0e-6);
        check_rdm(14, {0, 1, 2, 3, 4, 5}, fermionic, false, 2.0e-6);
        check_rdm(14, {0, 1, 2}, fermionic, true, 1.0e-12);
        check_rdm(14, {0, 1, 2, 3, 4, 5}, fermionic, true, 1.0e-12);
    }
    check_fermionic_trace_is_live(14, {3, 7, 11});
    check_fermionic_trace_is_live(14, {1, 3, 5, 7, 9, 11});

    std::printf("\nHalf-cut entropy (Gram + heevd vs fp64 reference, bits):\n");
    for (bool fermionic : {false, true})
    {
        check_entropy<32>(10, {0, 1, 2, 3, 4}, fermionic, false, 5.0e-4);
        check_entropy<64>(12, {0, 1, 2, 3, 4, 5}, fermionic, false, 5.0e-4);
        check_entropy<64>(12, {0, 2, 4, 6, 8, 10}, fermionic, false, 5.0e-4);
        check_entropy<64>(12, {0, 1, 2, 3, 4, 5}, fermionic, true, 1.0e-9);
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "\n%d failure(s).\n", failures);
        return 1;
    }
    std::printf("\nAll RDM/Gram tests passed.\n");
    return 0;
}

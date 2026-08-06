// Mode-3 (anisotropy) equivalence harness: CUDA-Q reference vs the cuStateVec
// engine, for the one circuit pattern cusv_equiv.cpp does not exercise.
//
// cusv_equiv runs every circuit on a register of exactly N qubits with nothing
// else in it. Mode 3 does not: probed::initialize_state allocates N+2 qubits up
// front (attach_at_start is false for two probes), evolves the *system* sites
// [0,N) for 4N timesteps while the two reference qubits [N,N+2) sit idle in
// |0>, then resets/entangles them mid-circuit and keeps evolving. So every
// cuStateVec op in mode 3 runs against a register two qubits wider than its
// own site range, and every measurement layer collapses a state in which the
// idle-or-entangled references are spectators.
//
// That is a distinct code path in two places:
//   * Engine::adopt() infers total_qubits from the CUDA-Q buffer size, so the
//     strides every apply/collapse uses are set by N+2, not N. A convention
//     mismatch between CUDA-Q's qubit ordering and the engine's bit ordering
//     is invisible at total==N (a global site relabelling is just a reflection
//     of a periodic chain) but not at total==N+2, where it would slide the
//     circuit onto the reference qubits.
//   * measure_layer() collapses sites in [0,N) on a 2^(N+2) vector, so its
//     joint-distribution reduction runs over an environment that includes the
//     references.
//
// Part 1 builds a real mode-3 prefix once -- advance, attach, advance, attach,
// at p>0 so the state is generic and the references are genuinely entangled --
// snapshots it, and then evolves that one snapshot through both backends at
// p=0, comparing state vectors amplitude by amplitude plus the two-probe mutual
// information computed independently on the host.
//
// The prefix is shared rather than replayed because probed::attach_reference
// calls reset() on a site that is entangled with the rest of the register.
// reset is a measurement: the runtime draws an outcome and the surviving branch
// depends on it, so replaying the schedule is not reproducible even within one
// backend (the isolation harness measures the same backend disagreeing with
// itself by ~7e-2). Branching from a common snapshot removes that draw from the
// comparison and leaves exactly the thing under test -- evolution driven on a
// register two qubits wider than the circuit, with the references entangled.
//
// Part 2 checks the batched collapse against a sequential Born reference on the
// wide register, with the references entangled so they are not a trivial
// tensor factor.
//
// Usage: ./mode3_equiv [n] [seed]

#include "mipt/circuit.hpp"
#include "mipt/cusv/engine.hpp"
#include "mipt/probed.hpp"

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
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

// The bare N+2 product register mode 3 starts from
// (probed::InitializeProductKernel).
struct ProductKernel
{
    void operator()(int total_qubits) __qpu__ { cudaq::qvector q(total_qubits); }
};

// Byte-for-byte the production probed::AttachReferenceKernel.
struct AttachKernel
{
    void operator()(cudaq::state *state, int n, int site, int reference) __qpu__
    {
        cudaq::qvector q(state);
        reset(q[site]);
        h(q[site]);
        cx(q[site], q[n + reference]);
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

// ---------------------------------------------------------------- host readout
//
// An independent two-probe readout, deliberately not the production
// ancilla::two_probe_observables: if both backends were fed through the same
// reduction the test could only catch disagreements, not a shared error in how
// the references are located in the register.

struct ProbeInfo
{
    double mutual_information = 0.0;
    double joint_entropy = 0.0;
};

// Eigenvalues of a small Hermitian matrix by cyclic Jacobi, then -sum p ln p.
double von_neumann_entropy(std::vector<std::complex<double>> rho, int dim)
{
    auto at = [&](int i, int j) -> std::complex<double> & { return rho[i * dim + j]; };
    for (int sweep = 0; sweep < 64; ++sweep)
    {
        double off = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            for (int j = i + 1; j < dim; ++j) { off += std::norm(at(i, j)); }
        }
        if (off < 1e-30) { break; }
        for (int p = 0; p < dim; ++p)
        {
            for (int q = p + 1; q < dim; ++q)
            {
                const std::complex<double> apq = at(p, q);
                if (std::abs(apq) < 1e-18) { continue; }
                const double app = std::real(at(p, p));
                const double aqq = std::real(at(q, q));
                const double phi = std::arg(apq);
                const std::complex<double> phase{std::cos(phi), std::sin(phi)};
                const double theta = 0.5 * std::atan2(2.0 * std::abs(apq), app - aqq);
                const double c = std::cos(theta);
                const double s = std::sin(theta);
                for (int k = 0; k < dim; ++k)
                {
                    const std::complex<double> akp = at(k, p);
                    const std::complex<double> akq = at(k, q);
                    at(k, p) = c * akp + s * std::conj(phase) * akq;
                    at(k, q) = -s * phase * akp + c * akq;
                }
                for (int k = 0; k < dim; ++k)
                {
                    const std::complex<double> apk = at(p, k);
                    const std::complex<double> aqk = at(q, k);
                    at(p, k) = c * apk + s * phase * aqk;
                    at(q, k) = -s * std::conj(phase) * apk + c * aqk;
                }
            }
        }
    }
    double entropy = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        const double lambda = std::real(rho[i * dim + i]);
        if (lambda > 1e-14) { entropy -= lambda * std::log(lambda); }
    }
    return entropy;
}

// rho_AB on the two reference qubits n and n+1, traced over the system, and
// I(A:B) = S(A) + S(B) - S(AB) in nats. Ordinary (qubit) trace.
ProbeInfo host_two_probe_info(const std::vector<std::complex<double>> &psi, int n)
{
    const std::size_t system_dim = std::size_t{1} << n;
    std::vector<std::complex<double>> rho(16, {0.0, 0.0});
    for (int a = 0; a < 4; ++a)
    {
        const std::size_t offset_a = ((a & 1) ? (std::size_t{1} << n) : 0u) |
                                     ((a & 2) ? (std::size_t{1} << (n + 1)) : 0u);
        for (int b = 0; b < 4; ++b)
        {
            const std::size_t offset_b = ((b & 1) ? (std::size_t{1} << n) : 0u) |
                                         ((b & 2) ? (std::size_t{1} << (n + 1)) : 0u);
            std::complex<double> sum{0.0, 0.0};
            for (std::size_t s = 0; s < system_dim; ++s)
            {
                sum += psi[s | offset_a] * std::conj(psi[s | offset_b]);
            }
            rho[a * 4 + b] = sum;
        }
    }

    std::vector<std::complex<double>> rho_a(4, {0.0, 0.0});
    std::vector<std::complex<double>> rho_b(4, {0.0, 0.0});
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                rho_a[i * 2 + j] += rho[(i + 2 * k) * 4 + (j + 2 * k)];
                rho_b[i * 2 + j] += rho[(k + 2 * i) * 4 + (k + 2 * j)];
            }
        }
    }

    ProbeInfo out;
    out.joint_entropy = von_neumann_entropy(rho, 4);
    out.mutual_information =
        von_neumann_entropy(rho_a, 2) + von_neumann_entropy(rho_b, 2) - out.joint_entropy;
    return out;
}

// ---------------------------------------------------------------- part 1

// The build-precision buffer cudaq::state::from_data wants.
#if MIPT_CUDAQ_PRECISION == 64
using Amplitude = std::complex<double>;
#else
using Amplitude = std::complex<float>;
#endif

// Evolves a restored snapshot through the readout times on one backend and
// returns the final state, recording the two-probe information at each readout.
std::vector<std::complex<double>> run_readouts(mipt::probed::CircuitWorkspace1D &workspace, int n,
                                               const std::vector<Amplitude> &snapshot,
                                               int first_timestep,
                                               const std::vector<int> &readouts, bool use_cusv,
                                               std::vector<ProbeInfo> &info_out)
{
    auto state = cudaq::state::from_data(snapshot);
    int current_t = first_timestep;

    const std::size_t dim = std::size_t{1} << (n + 2);
    std::vector<std::complex<double>> host;
    info_out.clear();
    for (const int tau : readouts)
    {
        const int target = first_timestep + tau;
        const int count = target - current_t;
        if (count > 0)
        {
            if (use_cusv) { workspace.advance(state, current_t, count); }
            else { state = workspace.advance_via_cudaq(state, current_t, count); }
            current_t = target;
        }
        reference_to_host(state, dim, host);
        info_out.push_back(host_two_probe_info(host, n));
    }
    return host;
}

void run_mode3_case(mipt::CircuitType type, int n, int delta_x, int delta_t, unsigned seed,
                    bool fp64)
{
    const char *name = mipt::circuit_type_name(type).data();
    const int t_eq = 4;
    const std::vector<int> readouts{0, 1, 2, 3, 5, 8};
    const int prefix_steps = t_eq + delta_t;
    const int timesteps = prefix_steps + readouts.back() + 1;
    const int site0 = 0;
    const int site1 = (site0 + delta_x) % n;

    // Measurements during the prefix only. That makes the shared prefix a
    // generic monitored-circuit state with the references really entangled,
    // and leaves the compared window deterministic.
    mipt::probed::CircuitWorkspace1D workspace;
    workspace.seed(seed);
    workspace.prepare_with_probability(
        n, timesteps, [prefix_steps](int step) { return step < prefix_steps ? 0.17 : 0.0; }, type);

    // Prefix: exactly run_mode3_trajectory's schedule, run once.
    auto state = cudaq::get_state(ProductKernel{}, n + 2);
    workspace.advance(state, 0, t_eq);
    state = cudaq::get_state(AttachKernel{}, &state, n, site0, 0);
    if (delta_t > 0) { workspace.advance(state, t_eq, delta_t); }
    state = cudaq::get_state(AttachKernel{}, &state, n, site1, 1);

    std::vector<Amplitude> snapshot(std::size_t{1} << (n + 2));
    state.to_host(snapshot.data(), snapshot.size());

    std::vector<ProbeInfo> cusv_info;
    std::vector<ProbeInfo> cudaq_info;
    const auto mine =
        run_readouts(workspace, n, snapshot, prefix_steps, readouts, true, cusv_info);
    const auto reference =
        run_readouts(workspace, n, snapshot, prefix_steps, readouts, false, cudaq_info);

    const auto c = compare(reference, mine);
    double worst_mi = 0.0;
    for (std::size_t k = 0; k < cusv_info.size(); ++k)
    {
        worst_mi = std::max(worst_mi, std::abs(cusv_info[k].mutual_information -
                                               cudaq_info[k].mutual_information));
    }

    const char *branch = delta_t == 0 ? "spatial" : "temporal";
    std::printf("  %-24s n=%2d %-8s dx=%d dt=%d : max|diff|=%.3e  |<ref|new>|=%.12f  "
                "max|dI|=%.3e\n",
                name, n, branch, delta_x, delta_t, c.max_abs, c.overlap, worst_mi);
    std::printf("      I(R1:R2) cusv  =");
    for (const auto &i : cusv_info) { std::printf(" %.6f", i.mutual_information); }
    std::printf("\n      I(R1:R2) cudaq =");
    for (const auto &i : cudaq_info) { std::printf(" %.6f", i.mutual_information); }
    std::printf("\n");

    const double tol = fp64 ? 1e-11 : 2e-4;
    const std::string label = std::string(name) + " " + branch;
    check(c.max_abs < tol, label + " statevector matches CUDA-Q with references attached");
    check(std::abs(c.overlap - 1.0) < (fp64 ? 1e-12 : 1e-5), label + " overlap is 1");
    check(worst_mi < (fp64 ? 1e-10 : 1e-3), label + " two-probe mutual information matches");

    // Sharpest available check on the engine's bit convention. The references
    // are isolated: no gate in the circuit acts on qubits n or n+1, so across
    // a measurement-free window rho_AB cannot change at all (the same
    // observation mode 5's probes=2 protocol rests on). If the engine's site
    // indices slid by the two ancillas on the N+2 register, the circuit would
    // be driving the references and both of these would move.
    double worst_drift_cusv = 0.0;
    double worst_drift_cudaq = 0.0;
    for (std::size_t k = 1; k < cusv_info.size(); ++k)
    {
        worst_drift_cusv = std::max(
            {worst_drift_cusv,
             std::abs(cusv_info[k].mutual_information - cusv_info.front().mutual_information),
             std::abs(cusv_info[k].joint_entropy - cusv_info.front().joint_entropy)});
        worst_drift_cudaq = std::max(
            {worst_drift_cudaq,
             std::abs(cudaq_info[k].mutual_information - cudaq_info.front().mutual_information),
             std::abs(cudaq_info[k].joint_entropy - cudaq_info.front().joint_entropy)});
    }
    std::printf("      rho_AB drift under measurement-free evolution: cusv=%.3e cudaq=%.3e\n",
                worst_drift_cusv, worst_drift_cudaq);
    // Loose on purpose. The floor here is fp32 accumulation, and it is set by
    // the CUDA-Q side: MMS runs the full gate decomposition and drifts ~1e-5,
    // where the engine's fused 4x4 passes drift ~1e-7. A site-indexing error
    // would move rho_AB by O(0.1), so this still has three orders of headroom.
    const double drift_tol = fp64 ? 1e-10 : 1e-4;
    check(worst_drift_cusv < drift_tol, label + " cusv leaves the isolated references untouched");
    check(worst_drift_cudaq < drift_tol, label + " cudaq leaves the isolated references untouched");

    if (delta_t == 0)
    {
        // Spatial branch: both references attach at the same instant to
        // distinct sites, so they are exactly uncorrelated at tau=0.
        check(std::abs(cusv_info.front().mutual_information) < 1e-6,
              label + " I(R1:R2)=0 at tau=0");
    }
}

// ---------------------------------------------------------------- part 2

// Host reference: sequential single-site Born sampling with renormalisation
// over the *full* N+2 register.
double host_sequential_collapse(std::vector<std::complex<double>> &psi,
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
            if (((i & mask) != 0 ? 1 : 0) == outcomes[k]) { branch += w; }
        }
        surprisal += -std::log(branch / total);
        const double scale = 1.0 / std::sqrt(branch);
        for (std::uint64_t i = 0; i < psi.size(); ++i)
        {
            psi[i] = (((i & mask) != 0 ? 1 : 0) == outcomes[k]) ? psi[i] * scale
                                                                : std::complex<double>{0.0, 0.0};
        }
    }
    return surprisal;
}

// Collapse a measurement layer on the wide register, with both references
// entangled into the system so they are genuinely correlated spectators.
void run_wide_measurement_case(int n, unsigned seed, bool fp64)
{
    const int total = n + 2;
    auto state = cudaq::get_state(ProductKernel{}, total);

    mipt::probed::CircuitWorkspace1D workspace;
    workspace.seed(seed);
    workspace.prepare(n, 6, 0.0, mipt::CircuitType::Haar);
    workspace.advance(state, 0, 3);
    state = cudaq::get_state(AttachKernel{}, &state, n, 0, 0);
    state = cudaq::get_state(AttachKernel{}, &state, n, n / 2, 1);
    workspace.advance(state, 3, 3);

    const std::size_t dim = std::size_t{1} << total;
    std::vector<std::complex<double>> before;
    reference_to_host(state, dim, before);

    // Adopt the same buffer the production path adopts, then collapse a set of
    // system sites -- never the references, exactly as a real layer does.
    mipt::cusv::Engine engine(fp64);
    const auto tensor = state.get_tensor();
    engine.adopt(tensor.data, total);
    check(engine.qubits() == total, "engine adopts the full N+2 register");

    std::vector<int> sites;
    for (int s = 1; s < n; s += 2) { sites.push_back(s); }

    std::mt19937 draw_rng(seed + 1000u);
    std::vector<int> outcomes;
    const double batched = engine.measure_layer(sites, draw_rng, &outcomes);
    engine.synchronize();

    std::vector<std::complex<double>> after;
    reference_to_host(state, dim, after);

    std::vector<std::complex<double>> host = before;
    const double sequential = host_sequential_collapse(host, sites, outcomes);

    const auto c = compare(host, after);
    const auto info_batched = host_two_probe_info(after, n);
    const auto info_host = host_two_probe_info(host, n);
    std::printf("  wide collapse  n=%2d total=%2d sites=%zu : max|diff|=%.3e  surprisal "
                "batched=%.9f sequential=%.9f  |dI|=%.3e\n",
                n, total, sites.size(), c.max_abs, batched, sequential,
                std::abs(info_batched.mutual_information - info_host.mutual_information));

    const double tol = fp64 ? 1e-11 : 2e-4;
    check(c.max_abs < tol, "batched collapse on N+2 qubits matches sequential collapse");
    check(std::abs(batched - sequential) < (fp64 ? 1e-10 : 1e-3),
          "joint surprisal on N+2 qubits equals summed sequential conditionals");
    check(std::abs(engine.squared_norm() - 1.0) < (fp64 ? 1e-12 : 1e-5),
          "wide state renormalised after collapse");
    check(std::abs(info_batched.mutual_information - info_host.mutual_information) < 1e-6,
          "post-collapse two-probe information matches the sequential reference");
}

} // namespace

int main(int argc, char **argv)
{
    const int n = argc > 1 ? std::stoi(argv[1]) : 10;
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 2024u;

    if (argc > 99)
    {
        cudaq::get_state(KeepLayerHelpersAlive{}, 2, std::vector<mipt::MmsLayer>{},
                         std::vector<mipt::HaarLayer>{}, std::vector<mipt::RppuLayer>{},
                         std::vector<mipt::RfgsLayer>{});
    }

    const bool fp64 = MIPT_CUDAQ_PRECISION == 64;
    std::cout << "mode3_equiv (CUDA-Q build precision fp" << MIPT_CUDAQ_PRECISION << ")\n";

    if (!mipt::cusv::enabled())
    {
        std::cout << "  cuStateVec engine disabled (MIPT_CUSV=0); nothing to compare.\n";
        return 1;
    }

    try
    {
        std::cout << "-- part 1: mode-3 trajectory with mid-circuit reference attachment, p=0 --\n";
        for (const auto type : {mipt::CircuitType::MMS, mipt::CircuitType::Haar,
                                mipt::CircuitType::FermionRPPU, mipt::CircuitType::QubitRPPU})
        {
            run_mode3_case(type, n, n / 2, 0, seed, fp64);       // spatial branch
            run_mode3_case(type, n, 0, 3, seed + 7u, fp64);      // temporal branch
        }

        std::cout << "-- part 2: measurement collapse on the N+2 register --\n";
        run_wide_measurement_case(n, seed, fp64);
        run_wide_measurement_case(n, seed + 11u, fp64);
    }
    catch (const std::exception &e)
    {
        std::cout << "  EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    if (failures == 0)
    {
        std::cout << "mode3_equiv: PASS\n";
        return 0;
    }
    std::cout << "mode3_equiv: " << failures << " FAILURE(S)\n";
    return 1;
}

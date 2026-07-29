// Ground truth extractor.
//
// Applies the *existing* CUDA-Q decompositions for one RPPU bond, one Haar
// bond, and one MMS bond to each computational basis state of a 2-qubit
// register, and prints the resulting 4x4 matrix column by column. That matrix
// is exactly what the fused single-matrix path must reproduce.
//
// Column c of the printed matrix = state after applying the decomposition to
// |c>, with basis index bit 0 = qubit 0.

#include "mipt/circuits/fermion.hpp"
#include "mipt/circuits/haar.hpp"
#include "mipt/circuits/mms.hpp"

#include <cudaq.h>

#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace
{

// nvq++ registers every __qpu__ inline free function reachable from an included
// header and emits a reference to it even when unused, so the apply_*_layers
// helpers must be called from some kernel in this translation unit or the link
// fails with undefined init_func symbols. This kernel is never executed.
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

// Prepare |basis> on 2 qubits, then run one RPPU bond gate.
struct RppuBondColumn
{
    void operator()(int basis, const std::vector<mipt::RppuBondGate> &gates) __qpu__
    {
        cudaq::qvector q(2);
        if (basis & 1) { x(q[0]); }
        if (basis & 2) { x(q[1]); }
        for (std::size_t gi = 0; gi < gates.size(); ++gi)
        {
            const auto gate = gates[gi];
            cx(q[gate.q0], q[gate.q1]);
            r1(gate.odd.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.odd.theta, gate.odd.phi, gate.odd.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
            r1(gate.even.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.even.theta, gate.even.phi, gate.even.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
            cx(q[gate.q0], q[gate.q1]);
        }
    }
};

struct HaarBondColumn
{
    void operator()(int basis, const std::vector<mipt::HaarBondGate> &gates) __qpu__
    {
        cudaq::qvector q(2);
        if (basis & 1) { x(q[0]); }
        if (basis & 2) { x(q[1]); }
        for (std::size_t gi = 0; gi < gates.size(); ++gi)
        {
            const auto gate = gates[gi];
            r1(gate.b1.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.b1.theta, gate.b1.phi, gate.b1.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
            r1(gate.b0.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.b0.theta, gate.b0.phi, gate.b0.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
            ry<cudaq::ctrl>(gate.theta1, q[gate.q0], q[gate.q1]);
            x(q[gate.q0]);
            ry<cudaq::ctrl>(gate.theta0, q[gate.q0], q[gate.q1]);
            x(q[gate.q0]);
            r1(gate.a1.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.a1.theta, gate.a1.phi, gate.a1.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
            r1(gate.a0.global_phase, q[gate.q1]);
            u3<cudaq::ctrl>(gate.a0.theta, gate.a0.phi, gate.a0.lambda, q[gate.q1], q[gate.q0]);
            x(q[gate.q1]);
        }
    }
};

// MMS bond on sites (0,1), reproducing apply_mms_layers exactly for one bond.
struct MmsBondColumn
{
    void operator()(int basis, const std::vector<int> &rx_flags,
                    const std::vector<int> &ry_flags,
                    const std::vector<int> &rxy_flags) __qpu__
    {
        cudaq::qvector q(2);
        if (basis & 1) { x(q[0]); }
        if (basis & 2) { x(q[1]); }
        for (int s = 0; s < 2; ++s)
        {
            if (rx_flags[s]) { rx(1.57079632679489661923, q[s]); }
            if (ry_flags[s]) { ry(1.57079632679489661923, q[s]); }
            if (rxy_flags[s])
            {
                rz(-0.78539816339744830962, q[s]);
                rx(1.57079632679489661923, q[s]);
                rz(0.78539816339744830962, q[s]);
            }
        }
        h(q[0]); h(q[1]);
        cx(q[0], q[1]);
        rz(1.57079632679489661923, q[1]);
        cx(q[0], q[1]);
        h(q[0]); h(q[1]);
    }
};

void print_matrix(const char *label, const std::complex<double> m[4][4])
{
    std::printf("%s\n", label);
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            std::printf("%+.15e %+.15e ", m[r][c].real(), m[r][c].imag());
        }
        std::printf("\n");
    }
}

template <typename Kernel, typename... Args>
void collect(const char *label, Kernel &&kernel, Args &&...args)
{
    std::complex<double> m[4][4]{};
    for (int basis = 0; basis < 4; ++basis)
    {
        auto state = cudaq::get_state(kernel, basis, args...);
        std::vector<std::complex<double>> host(4);
        state.to_host(host.data(), host.size());
        for (int r = 0; r < 4; ++r) { m[r][basis] = host[r]; }
    }
    print_matrix(label, m);
}

} // namespace

int main(int argc, char **argv)
{
    const unsigned seed = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 12345u;
    std::mt19937 rng(seed);

    // Never true. Present only so KeepLayerHelpersAlive is ODR-used and the
    // apply_*_layers definitions are emitted into this translation unit.
    if (argc > 99)
    {
        cudaq::get_state(KeepLayerHelpersAlive{}, 2, std::vector<mipt::MmsLayer>{},
                         std::vector<mipt::HaarLayer>{}, std::vector<mipt::RppuLayer>{},
                         std::vector<mipt::RfgsLayer>{});
    }

    // RPPU: one adjacent parity-preserving bond on (0,1).
    {
        std::vector<mipt::RppuBondGate> gates{mipt::random_parity_preserving_gate(0, 1, rng)};
        std::printf("RPPU_PARAMS even %.17g %.17g %.17g %.17g odd %.17g %.17g %.17g %.17g\n",
                    gates[0].even.global_phase, gates[0].even.theta, gates[0].even.phi, gates[0].even.lambda,
                    gates[0].odd.global_phase, gates[0].odd.theta, gates[0].odd.phi, gates[0].odd.lambda);
        collect("RPPU_MATRIX", RppuBondColumn{}, gates);
    }

    // Haar: one U(4) bond on (0,1).
    {
        const auto u = mipt::haar_unitary_4(rng);
        std::printf("HAAR_SAMPLED\n");
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                std::printf("%+.15e %+.15e ", u[r * 4 + c].real(), u[r * 4 + c].imag());
            }
            std::printf("\n");
        }
        std::vector<mipt::HaarBondGate> gates{mipt::decompose_haar_u4_csd(0, 1, u)};
        collect("HAAR_MATRIX", HaarBondColumn{}, gates);
    }

    // MMS: all 9 rotation-axis combinations on the two endpoints.
    {
        for (int a0 = 0; a0 < 3; ++a0)
        {
            for (int a1 = 0; a1 < 3; ++a1)
            {
                std::vector<int> rxf{a0 == 0 ? 1 : 0, a1 == 0 ? 1 : 0};
                std::vector<int> ryf{a0 == 1 ? 1 : 0, a1 == 1 ? 1 : 0};
                std::vector<int> rxyf{a0 == 2 ? 1 : 0, a1 == 2 ? 1 : 0};
                char label[64];
                std::snprintf(label, sizeof(label), "MMS_MATRIX axis0=%d axis1=%d", a0, a1);
                collect(label, MmsBondColumn{}, rxf, ryf, rxyf);
            }
        }
    }

    return 0;
}

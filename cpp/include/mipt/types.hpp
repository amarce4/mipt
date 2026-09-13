#pragma once

#include "mipt/backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mipt
{
template <typename T>
std::vector<double> linspace(T first, T last, int count)
{
    if (count <= 0) return {};
    if (count == 1) return {static_cast<double>(first)};

    std::vector<double> values(static_cast<std::size_t>(count));
    const double begin = static_cast<double>(first);
    const double step = (static_cast<double>(last) - begin) / static_cast<double>(count - 1);
    for (int i = 0; i < count; ++i)
    {
        values[static_cast<std::size_t>(i)] = begin + step * static_cast<double>(i);
    }
    values.back() = static_cast<double>(last);
    return values;
}

struct U2Params
{
    // U = exp(i global_phase) * U3(theta, phi, lambda).
    double global_phase = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    double lambda = 0.0;
};
inline U2Params decompose_u2_to_u3_phase(const std::array<std::complex<double>, 4> &u)
{
    const auto a = u[0];
    const auto b = u[1];
    const auto c = u[2];
    const auto d = u[3];

    const double ca = std::abs(a);
    const double sa = std::abs(c);
    U2Params out;
    out.theta = 2.0 * std::atan2(sa, ca);

    constexpr double eps = 1e-12;
    if (ca > eps && sa > eps)
    {
        out.global_phase = std::arg(a);
        out.phi = std::arg(c) - out.global_phase;
        out.lambda = std::arg(-b) - out.global_phase;
    }
    else if (ca > eps)
    {
        out.global_phase = std::arg(a);
        out.phi = 0.0;
        out.lambda = std::arg(d) - out.global_phase;
    }
    else
    {
        out.global_phase = 0.0;
        out.phi = std::arg(c);
        out.lambda = std::arg(-b);
    }

    return out;
}

struct CircuitWorkStats
{
    std::size_t layers = 0;
    std::size_t measurements = 0;
    std::size_t logical_two_site_gates = 0;
    std::size_t mms_bonds = 0;
    std::size_t fermion_parity_gates = 0;
    std::size_t fermion_jw_boundary_gates = 0;
    std::size_t fermion_jw_cz_corrections = 0;
    std::size_t fswap_gates = 0;
    std::size_t rfgs_bonds = 0;
    std::size_t wrapping_rfgs_bonds = 0;
    std::size_t estimated_cudaq_two_qubit_ops = 0;
};

inline std::string circuit_work_stats_string(const char *label,
                                      const CircuitWorkStats &stats,
                                      int n,
                                      int periods,
                                      double p)
{
    std::ostringstream out;
    out << label
        << " stats: n=" << n
        << " periods=" << periods
        << " p=" << p
        << " layers=" << stats.layers
        << " measurements=" << stats.measurements
        << " logical_two_site_gates=" << stats.logical_two_site_gates
        << " estimated_cudaq_two_qubit_ops=" << stats.estimated_cudaq_two_qubit_ops;
    if (stats.mms_bonds)
    {
        out << " mms_bonds=" << stats.mms_bonds;
    }
    if (stats.fermion_parity_gates || stats.fswap_gates)
    {
        out << " parity_gates=" << stats.fermion_parity_gates
            << " fswap_gates=" << stats.fswap_gates;
        if (stats.fermion_jw_boundary_gates)
        {
            out << " jw_boundary_gates=" << stats.fermion_jw_boundary_gates
                << " jw_cz_corrections=" << stats.fermion_jw_cz_corrections;
        }
    }
    if (stats.rfgs_bonds)
    {
        out << " rfgs_bonds=" << stats.rfgs_bonds
            << " wrapping_rfgs_bonds=" << stats.wrapping_rfgs_bonds;
    }
    return out.str();
}

template <typename LayerVector>
void apply_debug_prefix_layer_limit(LayerVector &layers, const char *label)
{
    const long limit = mipt::backend::debug_prefix_layers();
    if (limit <= 0)
    {
        return;
    }
    const auto old_size = layers.size();
    if (old_size > static_cast<std::size_t>(limit))
    {
        layers.resize(static_cast<std::size_t>(limit));
    }
    std::cerr << "[mipt-debug] WARNING: MIPT_DEBUG_PREFIX_LAYERS=" << limit
              << " is set; " << label << " circuit was truncated from "
              << old_size << " to " << layers.size()
              << " layers for backend profiling only. Output is not a valid production MIPT sample.\n";
}


enum class CircuitType : int
{
    MMS = 0,
    Haar = 1,
    FermionRPPU = 2,
    RFGS = 3,
    QubitRPPU = 4,
};

struct CircuitInfo
{
    CircuitType type;
    std::string_view name;
    std::string_view tag;
    std::string_view help;
    bool fermionic_trace;
    bool requires_even_sites;
};

inline constexpr std::array<CircuitInfo, 5> CIRCUITS{{
    {CircuitType::MMS, "MMS", "mms", "MMS gate set", false, false},
    {CircuitType::Haar, "Random Haar Unitary", "haar",
     "random Haar U(4) brickwork circuit", false, true},
    {CircuitType::FermionRPPU,
     "Fermionic Random Parity Preserving Unitary", "rppu",
     "fermionic RPPU with fermionic partial tracing", true, true},
    {CircuitType::RFGS, "RFGS", "rfgs",
     "fermionic reduced gate set with fermionic partial tracing", true, true},
    {CircuitType::QubitRPPU,
     "qRPPU (qubit parity-preserving unitary)", "qrppu",
     "qRPPU without JW/CZ chains, using qubit partial tracing", false, true},
}};

inline const CircuitInfo &circuit_info(CircuitType type)
{
    const auto index = static_cast<std::size_t>(type);
    if (index >= CIRCUITS.size() || CIRCUITS[index].type != type)
    {
        throw std::invalid_argument("Unknown circuit type.");
    }
    return CIRCUITS[index];
}

inline std::string_view circuit_type_name(CircuitType type)
{
    return circuit_info(type).name;
}

inline std::string_view circuit_type_tag(CircuitType type)
{
    return circuit_info(type).tag;
}

inline std::string circuit_type_options()
{
    std::ostringstream out;
    for (std::size_t i = 0; i < CIRCUITS.size(); ++i)
    {
        if (i != 0)
        {
            out << (i + 1 == CIRCUITS.size() ? ", or " : ", ");
        }
        out << static_cast<int>(CIRCUITS[i].type) << '=' << CIRCUITS[i].name;
    }
    return out.str();
}

inline void print_circuit_type_help(std::ostream &out, std::string_view indent)
{
    for (const auto &info : CIRCUITS)
    {
        out << indent << static_cast<int>(info.type) << " = " << info.help << '\n';
    }
}

inline CircuitType parse_circuit_type(int value)
{
    if (value < 0 || value >= static_cast<int>(CIRCUITS.size()))
    {
        throw std::invalid_argument("circ_type must be one of " +
                                    circuit_type_options() + '.');
    }
    return static_cast<CircuitType>(value);
}

inline bool uses_fermionic_trace(CircuitType type)
{
    return circuit_info(type).fermionic_trace;
}

inline bool preserves_computational_parity(CircuitType type)
{
    return type == CircuitType::FermionRPPU ||
           type == CircuitType::RFGS ||
           type == CircuitType::QubitRPPU;
}

// A 64-bit stream splitter. splitmix64 is a bijection, so distinct
// (master, index) pairs give distinct seeds and the master seed recorded in a
// run's output is enough to replay any one of its trajectories. Lives here
// rather than in circuit.hpp so host-only code -- the pair-gap control
// selection, and its tests -- can derive the same streams without CUDA-Q.
// In their own namespace because free_energy.exe has an identical splitmix64 of
// its own (free_energy_resume.hpp) brought in by a using-declaration, and the
// two would otherwise be ambiguous there.
namespace seeding
{
inline std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

inline std::uint64_t trajectory_seed(std::uint64_t master, std::uint64_t index)
{
    return splitmix64(master + 0x2545f4914f6cdd1dull * (index + 1ull));
}
} // namespace seeding

// ---------------------------------------------------------------------------
// Logical layer description
//
// The *simulated* layer types (MmsLayer, HaarLayer, RppuLayer, RfgsLayer) carry
// everything a state vector needs and nothing a graph does: gate parameters,
// Jordan-Wigner corrections, decomposition scaffolding. A LogicalLayer is the
// projection of one of those onto the only two facts the spacetime percolation
// graph cares about -- which sites a layer entangles with each other, and which
// sites it measures.
//
// Three conventions, each of which is a modelling choice rather than a
// transcription, and each of which would silently change the graph if it were
// made differently:
//
//   * An FSWAP is *transport*, not an entangling seed. It exchanges two
//     fermionic modes exactly, so it permutes worldline labels and creates no
//     correlation. Emitting a bond for it would fuse every site the historical
//     RPPU boundary SWAP network passes through into one cluster, which is the
//     whole ring.
//
//   * A direct Jordan-Wigner boundary gate is the *single* logical bond
//     (0, L-1). The CZ string it carries touches every site in between, but
//     those are basis-change corrections that undo themselves; the gate
//     entangles its two endpoints and nothing else. Emitting a bond per crossed
//     site would, again, connect the whole ring.
//
//   * Consequently the FSWAP-network boundary and the JW-string boundary
//     produce *identical* logical layers, which is correct: they are two
//     implementations of one nonlocal gate. `boundary_implementation` is
//     recorded in the output so that identity can be checked rather than
//     assumed.
//
// `measured[m]` is indexed by logical mode, not by qubit position, so a layer
// that transports modes is still described in the labels its neighbours use.
// ---------------------------------------------------------------------------

struct LogicalBond
{
    int a = 0;
    int b = 0;
    // How much amplitude this gate actually moves in each parity sector, as
    // the squared off-diagonal weight of that sector's 2x2 block.
    //
    //   pair_weight (even block, |00> <-> |11>) carries the pairing channel F;
    //   hop_weight  (odd block,  |01> <-> |10>) carries the hopping channel G.
    //
    // The binary graph treats every gate as an equally usable edge, which is
    // exactly the approximation under suspicion: a gate whose odd block is
    // nearly diagonal percolates just as well as one that mixes maximally, yet
    // transmits almost no parity-odd coherence. 1.0 is "fully transmitting" and
    // is what a circuit with no parity-block structure reports, so the weighted
    // analysis then degenerates to the unweighted one rather than lying.
    double pair_weight = 1.0;
    double hop_weight = 1.0;
};

struct LogicalLayer
{
    std::vector<LogicalBond> bonds;
    std::vector<std::uint8_t> measured;
};

// One trajectory's worth of logical layers, plus the permutation that says
// which mode each qubit position holds at the end.
//
// For every circuit in the registry that permutation is the identity -- the
// only transport in the tree is the RPPU boundary SWAP network, which is
// balanced within its layer -- but it is tracked and reported rather than
// assumed, because a new circuit that ended on a net permutation would
// otherwise mislabel both endpoints of every pair silently.
struct LogicalHistory
{
    int n = 0;
    std::vector<LogicalLayer> layers;
    std::vector<int> mode_at_site; // qubit position -> logical mode, at the end
    bool valid = false;

    void reset(int sites)
    {
        n = sites;
        layers.clear();
        mode_at_site.resize(static_cast<std::size_t>(sites));
        for (int site = 0; site < sites; ++site)
        {
            mode_at_site[static_cast<std::size_t>(site)] = site;
        }
        valid = false;
    }

    bool identity_permutation() const
    {
        for (int site = 0; site < n; ++site)
        {
            if (mode_at_site[static_cast<std::size_t>(site)] != site)
            {
                return false;
            }
        }
        return true;
    }
};

// A brickwork bond sweep, shared by every circuit whose layer is "pair up
// sites from `start`, optionally wrapping". Mirrors the bond loops in
// haar.hpp, mms.hpp and fermion.hpp rather than reimplementing them.
inline void append_brickwork_bonds(LogicalLayer &layer, int n, int start, bool wraps,
                                   const std::vector<int> &mode_at_site)
{
    const int bond_stop = (wraps && start == 1 && n > 2) ? n : (n - 1);
    for (int i = start; i < bond_stop; i += 2)
    {
        const int j = (i + 1) % n;
        layer.bonds.push_back({mode_at_site[static_cast<std::size_t>(i)],
                               mode_at_site[static_cast<std::size_t>(j)]});
    }
}

inline void set_measured(LogicalLayer &layer, int n, const std::vector<int> &measure_flags,
                         const std::vector<int> &mode_at_site)
{
    layer.measured.assign(static_cast<std::size_t>(n), 0u);
    for (int site = 0; site < n; ++site)
    {
        if (measure_flags[static_cast<std::size_t>(site)])
        {
            layer.measured[static_cast<std::size_t>(
                mode_at_site[static_cast<std::size_t>(site)])] = 1u;
        }
    }
}

inline bool requires_even_sites(CircuitType type)
{
    return circuit_info(type).requires_even_sites;
}

inline void validate_circuit_site_count(CircuitType type, int sites,
                                        std::string_view site_name = "n")
{
    if (requires_even_sites(type) && (sites % 2 != 0))
    {
        throw std::invalid_argument(std::string(circuit_type_name(type)) +
                                    " requires an even " + std::string(site_name) +
                                    " for periodic brickwork geometry.");
    }
}

} // namespace mipt

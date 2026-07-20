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

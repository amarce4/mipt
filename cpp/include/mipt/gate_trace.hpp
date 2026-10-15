#pragma once

// Stage 2 of the gate-path diagnostics: what the gates on a pair's connecting
// paths actually *did* to the state, in the trajectory that was measured.
//
// The question
// ------------
// Stage 1 (dist_connectivity.hpp, GatePathMetrics) says how entangling the
// gates on a connected-but-unentangled pair's best paths *could* be -- a
// property of the drawn unitaries alone. It cannot say whether those gates
// had anything to work on. A maximally entangling gate applied to two modes
// that are each maximally mixed, or that a measurement in the same layer
// resets, transmits nothing; a weak gate applied to a coherent pair can still
// matter. Stage 2 answers that per gate: rppu_gate_trace.exe regenerates the
// trajectory from the record file's master seed, replays it through the same
// cuStateVec engine calls production made (circuit.hpp's LayerObserver), and at
// every gate on the pair's shortest-maximin and widest paths reports the
// two-mode fermionic state of the gate's own modes before the gate, after its
// unitary, and after the layer's measurements.
//
// What lives here
// ---------------
// Everything that can be decided on the host: the parity-sector expansion of a
// snapshot, the two-mode fermionic reduction, the conjugation by a gate, the
// per-state metrics, the selection CSV, the validation logic that compares a
// replay against the stored records, and the two output schemas. The app is
// orchestration plus the GPU. `make test-dist` pins this header
// (tests/gate_trace_tests.cpp) against closed forms and first principles.
//
// Why `U rho U^dag` is the post-unitary state, exactly
// ----------------------------------------------------
// The snapshot is taken at the layer boundary, where every circuit in the
// registry has the identity mode permutation (the FSWAP network walks the
// boundary mode out and back inside one layer). Within a layer each mode is
// touched by at most one gate, and every other gate is parity-even on modes
// disjoint from (a, b), so the two-mode fermionic RDM after *all* of a layer's
// unitaries is the gate's U conjugating the RDM before them. U is
// rppu_gate_matrix(gate) with bit 0 = the gate's q0 = the mode bond.a holds,
// which is exactly the RDM ordering reduce_pair(psi, n, a, b) uses (bit 0 =
// first retained mode). That this holds for the adjacent gates, for the
// Jordan-Wigner boundary gate, *and* for the FSWAP-network boundary gate is
// not argued here but tested, on full host states, in gate_trace_tests.cpp.

#include "mipt/analysis/gate_entanglement.hpp"
#include "mipt/dist_connectivity.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/dist_record_reader.hpp"
#include "mipt/dist_records.hpp"
#include "mipt/dist_scaling_csv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mipt::gate_trace
{

using Complex = std::complex<double>;
using Rho4 = std::array<Complex, 16>; // row-major, basis index n_a + 2 n_b

inline constexpr int GATE_TRACE_FORMAT_VERSION = 1;

// The validation tolerances. The replay runs the same calls on the same build,
// so in practice it agrees bit for bit; these are what "the same trajectory"
// is allowed to mean when it does not (a different GPU reducing in a different
// order). fN gets the absolute floor of the closed form's own resolution plus
// a relative term; everything else a generic one.
inline constexpr double FMN_ATOL = 1.0e-13;
inline constexpr double FMN_RTOL = 1.0e-10;
inline constexpr double VALUE_ATOL = 1.0e-12;
inline constexpr double VALUE_RTOL = 1.0e-9;
// Host reduction of the observed snapshot against the production GPU
// reduction of the final state, entrywise. Both are fp64 sums of the same
// products in different orders.
inline constexpr double ANCHOR_RDM_TOL = 1.0e-12;

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

// The full 2^n state from the even-parity-sector encoding the engine runs on.
//
// One definition, cusv/parity.hpp's: an encoded index e carries b_0..b_{n-2}
// and the missing bit is b_{n-1} = parity(e). So amplitude e lands at
// e | (parity(e) << (n-1)), and every other slot is zero -- which is also
// what the device kernel (src/cuda/cusv_parity.cu, parity_expand_kernel)
// writes.
inline std::vector<Complex> expand_parity_sector(const std::vector<Complex> &encoded, int n)
{
    if (n < 2 || encoded.size() != (std::size_t{1} << (n - 1)))
    {
        throw std::invalid_argument("expand_parity_sector: encoded state has the wrong size.");
    }
    const std::size_t half = std::size_t{1} << (n - 1);
    std::vector<Complex> full(half * 2u, Complex(0.0, 0.0));
    for (std::size_t e = 0; e < half; ++e)
    {
        const bool odd = (__builtin_popcountll(static_cast<unsigned long long>(e)) & 1) != 0;
        full[odd ? (e | half) : e] = encoded[e];
    }
    return full;
}

// The two-mode fermionic RDM of modes (a, b), bit 0 = a, normalized to unit
// trace.
//
// The same convention as dist_replay.hpp's reduce_modes and the CUDA kernels
// (the sign is the inversion count of the permutation taking the ascending
// occupied-mode list to (retained in order, environment ascending)), written
// in closed form so it costs one pass over the state rather than an O(n^2)
// permutation count per amplitude: moving a to the front crosses the occupied
// environment modes below a, moving b behind it crosses those below b, and if
// a > b the two retained modes cross each other once when both are occupied.
// gate_trace_tests.cpp checks it against reduce_modes.
inline Rho4 reduce_pair(const std::vector<Complex> &psi, int n, int a, int b)
{
    if (a == b || a < 0 || b < 0 || a >= n || b >= n || psi.size() != (std::size_t{1} << n))
    {
        throw std::invalid_argument("reduce_pair: invalid modes or state size.");
    }
    const std::size_t mask_a = std::size_t{1} << a;
    const std::size_t mask_b = std::size_t{1} << b;
    const std::size_t below_a = mask_a - 1u;
    const std::size_t below_b = mask_b - 1u;
    const int crossed = a > b ? 1 : 0;
    Rho4 rho{};
    Complex column[4];
    for (std::size_t env = 0; env < psi.size(); ++env)
    {
        if ((env & (mask_a | mask_b)) != 0u)
        {
            continue;
        }
        const int pa = __builtin_popcountll(static_cast<unsigned long long>(env & below_a)) & 1;
        const int pb = __builtin_popcountll(static_cast<unsigned long long>(env & below_b)) & 1;
        for (int r = 0; r < 4; ++r)
        {
            const int ra = r & 1;
            const int rb = (r >> 1) & 1;
            const std::size_t index = env | (ra ? mask_a : 0u) | (rb ? mask_b : 0u);
            const int sign = (ra & pa) ^ (rb & pb) ^ (crossed & ra & rb);
            column[r] = sign ? -psi[index] : psi[index];
        }
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                rho[static_cast<std::size_t>(r * 4 + c)] += column[r] * std::conj(column[c]);
            }
        }
    }
    double trace = 0.0;
    for (int d = 0; d < 4; ++d)
    {
        trace += rho[static_cast<std::size_t>(d * 5)].real();
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("reduce_pair: the state has no weight.");
    }
    for (Complex &value : rho)
    {
        value /= trace;
    }
    return rho;
}

// U rho U^dag.
inline Rho4 conjugate(const GateMatrix4 &u, const Rho4 &rho)
{
    Rho4 tmp{};
    Rho4 out{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < 4; ++k)
            {
                sum += u[static_cast<std::size_t>(r * 4 + k)] * rho[static_cast<std::size_t>(k * 4 + c)];
            }
            tmp[static_cast<std::size_t>(r * 4 + c)] = sum;
        }
    }
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < 4; ++k)
            {
                sum += tmp[static_cast<std::size_t>(r * 4 + k)] *
                       std::conj(u[static_cast<std::size_t>(c * 4 + k)]);
            }
            out[static_cast<std::size_t>(r * 4 + c)] = sum;
        }
    }
    return out;
}

// Projective occupation measurement of one of the two modes (slot 0 = a,
// slot 1 = b) with the given outcome, renormalized. A diagonal projector, so
// no Jordan-Wigner sign enters. Throws on an outcome of zero probability.
inline Rho4 measure_occupation(const Rho4 &rho, int slot, int outcome)
{
    Rho4 out{};
    double weight = 0.0;
    for (int r = 0; r < 4; ++r)
    {
        if (((r >> slot) & 1) != outcome)
        {
            continue;
        }
        weight += rho[static_cast<std::size_t>(r * 5)].real();
        for (int c = 0; c < 4; ++c)
        {
            if (((c >> slot) & 1) == outcome)
            {
                out[static_cast<std::size_t>(r * 4 + c)] = rho[static_cast<std::size_t>(r * 4 + c)];
            }
        }
    }
    if (!(weight > 0.0))
    {
        throw std::runtime_error("measure_occupation: outcome has zero probability.");
    }
    for (Complex &value : out)
    {
        value /= weight;
    }
    return out;
}

inline double frobenius_distance(const Rho4 &x, const Rho4 &y)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < 16; ++i)
    {
        sum += std::norm(x[i] - y[i]);
    }
    return std::sqrt(sum);
}

inline double max_abs_difference(const Rho4 &x, const Rho4 &y)
{
    double out = 0.0;
    for (std::size_t i = 0; i < 16; ++i)
    {
        out = std::max(out, std::abs(x[i] - y[i]));
    }
    return out;
}

// The interleaved (re, im) layout dist_metrics.hpp and the GPU reductions use.
inline std::array<double, 32> pack(const Rho4 &rho)
{
    std::array<double, 32> out{};
    for (std::size_t i = 0; i < 16; ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

inline Rho4 unpack(const double *rho_ri)
{
    Rho4 out{};
    for (std::size_t i = 0; i < 16; ++i)
    {
        out[i] = Complex(rho_ri[2 * i], rho_ri[2 * i + 1]);
    }
    return out;
}

// The quantities each trace row reports for one two-mode state. fmn/fmi/fg2/
// ff2 are the production ones (dist::two_party_metrics with the fermionic
// trace and the parity-preserving closed form -- the same call PairProtocol
// makes), so a trace value and a record value are the same computation.
struct PairState
{
    double fmn = std::numeric_limits<double>::quiet_NaN();
    double fmi = std::numeric_limits<double>::quiet_NaN();
    double fg2 = std::numeric_limits<double>::quiet_NaN();
    double ff2 = std::numeric_limits<double>::quiet_NaN();
    double i_occ = std::numeric_limits<double>::quiet_NaN();
    double purity = std::numeric_limits<double>::quiet_NaN();           // Tr rho^2
    double distance_from_I4 = std::numeric_limits<double>::quiet_NaN(); // ||rho - I/4||_F
};

inline PairState pair_state(const Rho4 &rho)
{
    const std::array<double, 32> packed = pack(rho);
    const dist::PairMetrics metrics = dist::two_party_metrics(packed.data(), true, true);
    PairState out;
    out.fmn = metrics.mn;
    out.fmi = metrics.mi;
    out.fg2 = metrics.g2;
    out.ff2 = metrics.f2;
    out.i_occ = metrics.i_occ;
    double purity = 0.0;
    Rho4 identity{};
    for (std::size_t i = 0; i < 16; ++i)
    {
        purity += std::norm(rho[i]); // Tr(rho rho^dag) = Tr rho^2 for Hermitian rho
    }
    for (int d = 0; d < 4; ++d)
    {
        identity[static_cast<std::size_t>(d * 5)] = Complex(0.25, 0.0);
    }
    out.purity = purity;
    out.distance_from_I4 = frobenius_distance(rho, identity);
    return out;
}

// ---------------------------------------------------------------------------
// The selection CSV (data_analysis/select_gate_traces.py)
// ---------------------------------------------------------------------------

inline constexpr const char *SELECTION_HEADER =
    "realization_id,pair_i,pair_j,group,geometry_id,embedding_id,fmn,fmi,shortest_path,"
    "component_size,widest_gate_ep";

inline bool valid_group(const std::string &group)
{
    return group == "silent" || group == "entangled_control" ||
           group == "coherent_subthreshold" || group == "occupation_correlated";
}

struct Selection
{
    std::uint32_t realization = 0;
    int i = 0;
    int j = 0;
    std::string group;
    int geometry_id = -1;
    int embedding_id = -1;
    double fmn = std::numeric_limits<double>::quiet_NaN();
    double fmi = std::numeric_limits<double>::quiet_NaN();
    double shortest_path = std::numeric_limits<double>::quiet_NaN();
    double component_size = std::numeric_limits<double>::quiet_NaN();
    double widest_gate_ep = std::numeric_limits<double>::quiet_NaN();
};

namespace detail
{
inline std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (std::size_t k = 0; k < line.size(); ++k)
    {
        const char ch = line[k];
        if (quoted)
        {
            if (ch == '"' && k + 1 < line.size() && line[k + 1] == '"')
            {
                cell += '"';
                ++k;
            }
            else if (ch == '"')
            {
                quoted = false;
            }
            else
            {
                cell += ch;
            }
        }
        else if (ch == '"')
        {
            quoted = true;
        }
        else if (ch == ',')
        {
            out.push_back(cell);
            cell.clear();
        }
        else if (ch != '\r')
        {
            cell += ch;
        }
    }
    out.push_back(cell);
    return out;
}

inline double parse_real(const std::string &text)
{
    if (text.empty() || text == "nan" || text == "NaN" || text == "NAN")
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size())
    {
        throw std::invalid_argument("not a number: `" + text + "`");
    }
    return value;
}

inline long parse_integer(const std::string &text)
{
    std::size_t used = 0;
    const long value = std::stol(text, &used);
    if (used != text.size())
    {
        throw std::invalid_argument("not an integer: `" + text + "`");
    }
    return value;
}
} // namespace detail

// Columns are located by name, so a reordered file still reads; but every
// contract column has to be there, and every group has to be one the selector
// defines -- a typo there would otherwise silently become its own group.
inline std::vector<Selection> read_selection(std::istream &in, const std::string &label)
{
    std::string line;
    if (!std::getline(in, line))
    {
        throw std::runtime_error(label + " is empty; expected the header `" +
                                 std::string(SELECTION_HEADER) + "`.");
    }
    const std::vector<std::string> header = detail::split_csv_line(line);
    auto column = [&](const char *name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end())
        {
            throw std::runtime_error(label + " has no `" + name + "` column; expected `" +
                                     std::string(SELECTION_HEADER) + "`.");
        }
        return static_cast<std::size_t>(it - header.begin());
    };
    const std::size_t c_real = column("realization_id");
    const std::size_t c_i = column("pair_i");
    const std::size_t c_j = column("pair_j");
    const std::size_t c_group = column("group");
    const std::size_t c_geometry = column("geometry_id");
    const std::size_t c_embedding = column("embedding_id");
    const std::size_t c_fmn = column("fmn");
    const std::size_t c_fmi = column("fmi");
    const std::size_t c_path = column("shortest_path");
    const std::size_t c_component = column("component_size");
    const std::size_t c_widest = column("widest_gate_ep");

    std::vector<Selection> out;
    std::size_t line_number = 1;
    while (std::getline(in, line))
    {
        ++line_number;
        if (line.empty() || line == "\r")
        {
            continue;
        }
        const std::vector<std::string> cells = detail::split_csv_line(line);
        if (cells.size() != header.size())
        {
            throw std::runtime_error(label + " line " + std::to_string(line_number) + " has " +
                                     std::to_string(cells.size()) + " cells; the header has " +
                                     std::to_string(header.size()) + ".");
        }
        try
        {
            Selection row;
            const long realization = detail::parse_integer(cells[c_real]);
            if (realization < 0 || realization > 0xffffffffL)
            {
                throw std::invalid_argument("realization_id out of range");
            }
            row.realization = static_cast<std::uint32_t>(realization);
            row.i = static_cast<int>(detail::parse_integer(cells[c_i]));
            row.j = static_cast<int>(detail::parse_integer(cells[c_j]));
            row.group = cells[c_group];
            row.geometry_id = static_cast<int>(detail::parse_integer(cells[c_geometry]));
            row.embedding_id = static_cast<int>(detail::parse_integer(cells[c_embedding]));
            row.fmn = detail::parse_real(cells[c_fmn]);
            row.fmi = detail::parse_real(cells[c_fmi]);
            row.shortest_path = detail::parse_real(cells[c_path]);
            row.component_size = detail::parse_real(cells[c_component]);
            row.widest_gate_ep = detail::parse_real(cells[c_widest]);
            if (!valid_group(row.group))
            {
                throw std::invalid_argument("unknown group `" + row.group + "`");
            }
            if (row.i == row.j || row.i < 0 || row.j < 0)
            {
                throw std::invalid_argument("invalid pair");
            }
            out.push_back(std::move(row));
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error(label + " line " + std::to_string(line_number) + ": " +
                                     error.what() + ".");
        }
    }
    return out;
}

inline std::vector<Selection> read_selection(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Could not open selection CSV " + path + ".");
    }
    return read_selection(in, path);
}

// ---------------------------------------------------------------------------
// Validation against the stored records
// ---------------------------------------------------------------------------

// The value a record named `name` would carry for this pair, recomputed from
// the replay -- a by-name mirror of PairProtocol::fill_record_values, with the
// gate-path values going through records::fill_gate_record_values, the very
// function production writes them with. An observable this does not know is a
// refusal: a replay that skipped a stored column would be validated against
// less than the file holds and would not say so.
inline std::vector<double> replayed_observables(const std::vector<std::string> &names,
                                                const dist::PairMetrics &metrics,
                                                const dist::PairMetrics &fermion,
                                                const dist::PairConnectivity &conn,
                                                const dist::GatePathMetrics &gate)
{
    std::array<double, dist::records::GATE_RECORD_VALUES> gate_values{};
    dist::records::fill_gate_record_values(conn, gate, gate_values.data());
    const auto &gate_names = dist::records::gate_observable_names();

    std::vector<double> out;
    out.reserve(names.size());
    for (const std::string &name : names)
    {
        const auto gate_it = std::find_if(gate_names.begin(), gate_names.end(),
                                          [&](const char *g) { return name == g; });
        if (gate_it != gate_names.end())
        {
            out.push_back(gate_values[static_cast<std::size_t>(gate_it - gate_names.begin())]);
        }
        else if (name == "mi") { out.push_back(metrics.mi); }
        else if (name == "mn") { out.push_back(metrics.mn); }
        else if (name == "fmi") { out.push_back(fermion.mi); }
        else if (name == "fmn") { out.push_back(fermion.mn); }
        else if (name == "n_i") { out.push_back(metrics.n_i); }
        else if (name == "n_j") { out.push_back(metrics.n_j); }
        else if (name == "dnn") { out.push_back(metrics.dnn); }
        else if (name == "g2") { out.push_back(metrics.g2); }
        else if (name == "f2") { out.push_back(metrics.f2); }
        else if (name == "fg2") { out.push_back(fermion.g2); }
        else if (name == "ff2") { out.push_back(fermion.f2); }
        // `full` detail: the channel phases, of the fermionic channel wherever
        // it exists (prefixed `f`), of the ordinary one otherwise.
        else if (name == "fre_g") { out.push_back(fermion.g.real()); }
        else if (name == "fim_g") { out.push_back(fermion.g.imag()); }
        else if (name == "fre_f") { out.push_back(fermion.f.real()); }
        else if (name == "fim_f") { out.push_back(fermion.f.imag()); }
        else if (name == "re_g") { out.push_back(metrics.g.real()); }
        else if (name == "im_g") { out.push_back(metrics.g.imag()); }
        else if (name == "re_f") { out.push_back(metrics.f.real()); }
        else if (name == "im_f") { out.push_back(metrics.f.imag()); }
        else
        {
            throw std::runtime_error("The gate-trace replay does not know how to recompute the "
                                     "record observable `" + name + "`, so it cannot validate "
                                     "a replay against this file.");
        }
    }
    return out;
}

// Whether a stored and a replayed value agree. NaN agrees only with NaN: a
// NaN in a record means "no path" or "not computed", and a replay that found
// a value there is a different trajectory.
inline bool values_agree(const std::string &name, double stored, double replayed)
{
    const bool stored_nan = std::isnan(stored);
    const bool replayed_nan = std::isnan(replayed);
    if (stored_nan || replayed_nan)
    {
        return stored_nan && replayed_nan;
    }
    const double scale = std::max(std::abs(stored), std::abs(replayed));
    const double diff = std::abs(stored - replayed);
    if (name == "fmn")
    {
        return diff <= FMN_ATOL + FMN_RTOL * scale;
    }
    return diff <= VALUE_ATOL + VALUE_RTOL * scale;
}

// One pair's replayed record, keyed the way the stored one is.
struct ReplayedRecord
{
    std::uint32_t flags = 0;
    std::vector<double> values;
};

struct RealizationCheck
{
    std::size_t records_compared = 0;
    std::size_t values_compared = 0;
    std::size_t mismatches = 0;
    std::size_t flag_mismatches = 0;
    double max_abs_error = 0.0;     // over every finite value compared
    double max_fmn_error = 0.0;     // over every pair's fN
    std::string first_mismatch;     // empty when everything agreed
    bool valid() const { return records_compared > 0 && mismatches == 0 && flag_mismatches == 0; }
};

// Compare every stored record of one realization against the replay.
//
// `replayed` is keyed by the ascending site pair. Every stored record must
// find its pair through the embedding table, every replayed pair must be
// stored exactly once (a record file at stride 1 holds every pair of a kept
// trajectory), and the flags must agree bit for bit -- the connectivity bits
// are a pure function of the layer draw and the positivity bits of values that
// agree, so a flag that differs is a different trajectory.
inline RealizationCheck check_realization(
    const std::vector<std::string> &observables,
    const std::vector<dist::records::StoredRecord> &stored,
    const std::map<std::pair<int, int>, std::vector<int>> &embedding_sites,
    const std::map<std::pair<int, int>, ReplayedRecord> &replayed)
{
    RealizationCheck check;
    auto note = [&](const std::string &what) {
        if (check.first_mismatch.empty())
        {
            check.first_mismatch = what;
        }
    };
    const auto fmn_slot = std::find(observables.begin(), observables.end(), "fmn");
    std::set<std::pair<int, int>> seen;
    for (const dist::records::StoredRecord &record : stored)
    {
        const auto sites = embedding_sites.find({record.geometry, record.embedding});
        if (sites == embedding_sites.end() || sites->second.size() != 2)
        {
            ++check.mismatches;
            note("record (" + std::to_string(record.geometry) + "," +
                 std::to_string(record.embedding) + ") is not in the embedding table");
            continue;
        }
        const std::pair<int, int> key{std::min(sites->second[0], sites->second[1]),
                                      std::max(sites->second[0], sites->second[1])};
        const auto match = replayed.find(key);
        if (match == replayed.end() || !seen.insert(key).second)
        {
            ++check.mismatches;
            note("pair (" + std::to_string(key.first) + "," + std::to_string(key.second) +
                 ") stored twice or not replayed");
            continue;
        }
        ++check.records_compared;
        if (record.flags != match->second.flags)
        {
            ++check.flag_mismatches;
            note("pair (" + std::to_string(key.first) + "," + std::to_string(key.second) +
                 ") flags stored " + std::to_string(record.flags) + " replayed " +
                 std::to_string(match->second.flags));
        }
        for (std::size_t slot = 0; slot < observables.size(); ++slot)
        {
            const double a = record.values[slot];
            const double b = match->second.values[slot];
            ++check.values_compared;
            if (std::isfinite(a) && std::isfinite(b))
            {
                check.max_abs_error = std::max(check.max_abs_error, std::abs(a - b));
                if (fmn_slot != observables.end() &&
                    slot == static_cast<std::size_t>(fmn_slot - observables.begin()))
                {
                    check.max_fmn_error = std::max(check.max_fmn_error, std::abs(a - b));
                }
            }
            if (!values_agree(observables[slot], a, b))
            {
                ++check.mismatches;
                std::ostringstream what;
                what.precision(17);
                what << "pair (" << key.first << "," << key.second << ") " << observables[slot]
                     << " stored " << a << " replayed " << b;
                note(what.str());
            }
        }
    }
    if (seen.size() != replayed.size())
    {
        ++check.mismatches;
        note(std::to_string(replayed.size() - seen.size()) +
             " replayed pair(s) have no stored record");
    }
    return check;
}

// ---------------------------------------------------------------------------
// Output schemas
// ---------------------------------------------------------------------------

enum class PathRole
{
    ShortestMaximin,
    Widest,
};

inline const char *path_role_name(PathRole role)
{
    return role == PathRole::ShortestMaximin ? "shortest_maximin" : "widest_entangling_power";
}

inline constexpr const char *TRACE_HEADER =
    "realization_id,pair_i,pair_j,group,path_role,layer,gate_mode_a,gate_mode_b,"
    "gate_entangling_power,even_block_mixing,odd_block_mixing,"
    "pre_gate_fmn,pre_gate_fmi,pre_gate_fg2,pre_gate_ff2,pre_gate_i_occ,pre_gate_pair_purity,"
    "pre_gate_distance_from_I4,"
    "post_unitary_fmn,post_unitary_fmi,post_unitary_fg2,post_unitary_ff2,rho_change_frobenius,"
    "post_measurement_fmn,post_measurement_fmi,post_measurement_fg2,post_measurement_ff2,"
    "final_anchor_fmn,final_anchor_fmi,"
    "path_position,edge_id,source_gate_index,gate_measured_a,gate_measured_b,trace_valid,"
    "validation_max_error";

struct TraceRow
{
    std::uint32_t realization = 0;
    int pair_i = 0;
    int pair_j = 0;
    std::string group;
    PathRole role = PathRole::ShortestMaximin;
    int layer = 0;
    int mode_a = 0;
    int mode_b = 0;
    double entangling_power = std::numeric_limits<double>::quiet_NaN();
    double even_block_mixing = std::numeric_limits<double>::quiet_NaN();
    double odd_block_mixing = std::numeric_limits<double>::quiet_NaN();
    PairState pre;
    PairState post_unitary;
    double rho_change_frobenius = std::numeric_limits<double>::quiet_NaN();
    PairState post_measurement;
    double final_anchor_fmn = std::numeric_limits<double>::quiet_NaN();
    double final_anchor_fmi = std::numeric_limits<double>::quiet_NaN();
    int path_position = 0;
    std::uint32_t edge_id = 0;
    int source_gate_index = -1;
    int measured_a = 0;
    int measured_b = 0;
    bool trace_valid = false;
    double validation_max_error = std::numeric_limits<double>::quiet_NaN();
};

inline std::string render_trace_row(const TraceRow &row)
{
    using dist::append_double;
    std::string out;
    out += std::to_string(row.realization);
    out += ',' + std::to_string(row.pair_i) + ',' + std::to_string(row.pair_j) + ',';
    out += row.group;
    out += ',';
    out += path_role_name(row.role);
    out += ',' + std::to_string(row.layer) + ',' + std::to_string(row.mode_a) + ',' +
           std::to_string(row.mode_b);
    for (const double value :
         {row.entangling_power, row.even_block_mixing, row.odd_block_mixing, row.pre.fmn,
          row.pre.fmi, row.pre.fg2, row.pre.ff2, row.pre.i_occ, row.pre.purity,
          row.pre.distance_from_I4, row.post_unitary.fmn, row.post_unitary.fmi,
          row.post_unitary.fg2, row.post_unitary.ff2, row.rho_change_frobenius,
          row.post_measurement.fmn, row.post_measurement.fmi, row.post_measurement.fg2,
          row.post_measurement.ff2, row.final_anchor_fmn, row.final_anchor_fmi})
    {
        out += ',';
        append_double(out, value);
    }
    out += ',' + std::to_string(row.path_position) + ',' + std::to_string(row.edge_id) + ',' +
           std::to_string(row.source_gate_index) + ',' + std::to_string(row.measured_a) + ',' +
           std::to_string(row.measured_b) + ',' + (row.trace_valid ? "1" : "0") + ',';
    append_double(out, row.validation_max_error);
    out += '\n';
    return out;
}

inline constexpr const char *VALIDATION_HEADER =
    "realization_id,pair_i,pair_j,group,engine_path,stored_fmn,replayed_fmn,stored_fmi,"
    "replayed_fmi,selection_fmn,selection_fmi,connected,gate_path_available,"
    "shortest_maximin_unitary_edges,widest_unitary_edges,records_compared,values_compared,"
    "mismatches,flag_mismatches,max_abs_error_all_pairs,max_fmn_error_all_pairs,"
    "anchor_host_gpu_rdm_max_diff,first_mismatch,valid";

struct ValidationRow
{
    std::uint32_t realization = 0;
    int pair_i = 0;
    int pair_j = 0;
    std::string group;
    std::string engine_path;
    double stored_fmn = std::numeric_limits<double>::quiet_NaN();
    double replayed_fmn = std::numeric_limits<double>::quiet_NaN();
    double stored_fmi = std::numeric_limits<double>::quiet_NaN();
    double replayed_fmi = std::numeric_limits<double>::quiet_NaN();
    double selection_fmn = std::numeric_limits<double>::quiet_NaN();
    double selection_fmi = std::numeric_limits<double>::quiet_NaN();
    bool connected = false;
    bool gate_path_available = false;
    std::size_t shortest_maximin_edges = 0;
    std::size_t widest_edges = 0;
    RealizationCheck check;
    double anchor_rdm_diff = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

inline std::string render_validation_row(const ValidationRow &row)
{
    using dist::append_double;
    std::string out;
    out += std::to_string(row.realization);
    out += ',' + std::to_string(row.pair_i) + ',' + std::to_string(row.pair_j) + ',';
    out += row.group;
    out += ',';
    out += row.engine_path;
    for (const double value : {row.stored_fmn, row.replayed_fmn, row.stored_fmi, row.replayed_fmi,
                               row.selection_fmn, row.selection_fmi})
    {
        out += ',';
        append_double(out, value);
    }
    out += ',';
    out += row.connected ? '1' : '0';
    out += ',';
    out += row.gate_path_available ? '1' : '0';
    out += ',' + std::to_string(row.shortest_maximin_edges) + ',' +
           std::to_string(row.widest_edges) + ',' + std::to_string(row.check.records_compared) +
           ',' + std::to_string(row.check.values_compared) + ',' +
           std::to_string(row.check.mismatches) + ',' + std::to_string(row.check.flag_mismatches) +
           ',';
    append_double(out, row.check.max_abs_error);
    out += ',';
    append_double(out, row.check.max_fmn_error);
    out += ',';
    append_double(out, row.anchor_rdm_diff);
    out += ',';
    out += dist::csv_quote(row.check.first_mismatch);
    out += ',';
    out += row.valid ? '1' : '0';
    out += '\n';
    return out;
}

// <selection stem>_gate_trace.csv and <selection stem>_gate_trace_validation.csv.
inline std::string default_trace_path(const std::string &selection_path)
{
    std::string stem = selection_path;
    const std::size_t slash = stem.find_last_of('/');
    const std::size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    {
        stem.erase(dot);
    }
    return stem + "_gate_trace.csv";
}

inline std::string validation_path_for(const std::string &trace_path)
{
    std::string stem = trace_path;
    const std::size_t slash = stem.find_last_of('/');
    const std::size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    {
        stem.erase(dot);
    }
    return stem + "_validation.csv";
}

} // namespace mipt::gate_trace

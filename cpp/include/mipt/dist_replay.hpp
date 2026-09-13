#pragma once

// Targeted high-precision replay of selected trajectories.
// --------------------------------------------------------
//
// **The question.** A connected pair whose fermionic negativity sits at 1e-16
// is either a structural zero, a roundoff artefact, or a real but weak signal,
// and no single-precision measurement can tell them apart. Replaying the same
// circuit at higher precision does: a structural zero stays exactly zero, a
// roundoff signal moves by roughly the precision change, and a physical weak
// signal converges.
//
// **Why the measurement record has to be forced.** A trajectory is a sequence
// of Born samples, each a threshold comparison against an accumulated
// probability. Change the arithmetic and some comparison eventually lands the
// other way, and from there the two runs are simulating *different*
// trajectories -- so their amplitudes differ by O(1) and the comparison says
// nothing about precision. The replay therefore samples the record once, at
// double, and then forces that exact record through the high-precision pass.
// Same circuit, same outcomes, different arithmetic: the only surviving
// difference is roundoff.
//
// **The circuit is regenerated, not stored.** `build_rppu_layers` is a pure
// function of (n, periods, p, rng), and the run's master seed plus the
// realization index reproduce the rng exactly -- the same
// `trajectory_seed(master, realization)` the production loop uses. So a replay
// needs only the ids, and it is guaranteed to be the same circuit the run
// measured rather than a fresh draw from the same ensemble.
//
// **The operator list is the production one.** `build_rppu_layer_ops` is what
// the cuStateVec engine executes; this replays that same list on a host state
// vector templated on the scalar type. There is no second definition of what an
// RPPU layer does, which is the only way the comparison means anything.
//
// What the extra precision actually buys: `long double` on x86-64 is the 80-bit
// format, a 64-bit mantissa against double's 53. That is ~3.3 extra decimal
// digits -- enough to separate 1e-16 roundoff from a signal, not enough to
// resolve 1e-30. It is reported as what it is.

#include "mipt/analysis/cut_negativity.hpp"
#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/cusv/gates.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/dist_scaling_csv.hpp"
#include "mipt/types.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace mipt::dist::replay
{

inline constexpr int REPLAY_FORMAT_VERSION = 1;

// A host state vector at an arbitrary scalar precision, executing the same
// operator list the device engine does.
template <typename Real>
class HostState
{
  public:
    explicit HostState(int qubits) : n_(qubits), amplitudes_(std::size_t{1} << qubits)
    {
        amplitudes_[0] = std::complex<Real>(1.0, 0.0);
    }

    int qubits() const { return n_; }
    const std::vector<std::complex<Real>> &amplitudes() const { return amplitudes_; }

    void apply(const cusv::Op &op)
    {
        switch (op.kind)
        {
        case cusv::OpKind::Matrix: apply_matrix(op); return;
        case cusv::OpKind::JwString: apply_jw_string(op); return;
        case cusv::OpKind::ParityGate:
            // Emitted only by the parity-sector rewriting, which runs on the
            // encoded 2^(n-1) picture. The replay executes the full 2^n
            // operator list, so reaching this is a wiring error rather than a
            // case to handle.
            throw std::runtime_error("Replay met a parity-sector operator on the full state.");
        }
    }

    // Born-sample or force the outcome of measuring `site`. Returns it.
    int measure(int site, int forced, Real uniform)
    {
        const std::size_t bit = std::size_t{1} << site;
        Real weight_one = 0;
        for (std::size_t index = 0; index < amplitudes_.size(); ++index)
        {
            if ((index & bit) != 0u)
            {
                weight_one += std::norm(amplitudes_[index]);
            }
        }
        const int outcome = forced >= 0 ? forced : (uniform < weight_one ? 1 : 0);
        const Real weight = outcome == 1 ? weight_one : Real(1) - weight_one;
        // A forced outcome of vanishing weight would divide by noise. It cannot
        // happen when the record came from this same circuit, so it is a
        // refusal rather than a clamp.
        if (!(weight > Real(1e-30)))
        {
            throw std::runtime_error("Replay forced a measurement outcome of zero probability.");
        }
        const Real scale = Real(1) / std::sqrt(weight);
        for (std::size_t index = 0; index < amplitudes_.size(); ++index)
        {
            const bool one = (index & bit) != 0u;
            amplitudes_[index] = (one == (outcome == 1))
                                     ? amplitudes_[index] * scale
                                     : std::complex<Real>(0, 0);
        }
        return outcome;
    }

  private:
    void apply_matrix(const cusv::Op &op)
    {
        const int k = static_cast<int>(op.targets.size());
        const std::size_t block = std::size_t{1} << k;
        std::vector<std::size_t> offsets(block);
        for (std::size_t sub = 0; sub < block; ++sub)
        {
            std::size_t offset = 0;
            for (int t = 0; t < k; ++t)
            {
                if ((sub >> t) & 1u)
                {
                    offset |= std::size_t{1} << op.targets[static_cast<std::size_t>(t)];
                }
            }
            offsets[sub] = offset;
        }
        std::size_t target_mask = 0;
        for (int t : op.targets)
        {
            target_mask |= std::size_t{1} << t;
        }
        std::vector<std::complex<Real>> gathered(block);
        std::vector<std::complex<Real>> result(block);
        for (std::size_t base = 0; base < amplitudes_.size(); ++base)
        {
            if ((base & target_mask) != 0u)
            {
                continue; // each block is visited once, from its zero corner
            }
            for (std::size_t sub = 0; sub < block; ++sub)
            {
                gathered[sub] = amplitudes_[base | offsets[sub]];
            }
            for (std::size_t r = 0; r < block; ++r)
            {
                std::complex<Real> sum(0, 0);
                for (std::size_t c = 0; c < block; ++c)
                {
                    const std::complex<double> m = op.matrix.at(r, c);
                    sum += std::complex<Real>(static_cast<Real>(m.real()),
                                              static_cast<Real>(m.imag())) *
                           gathered[c];
                }
                result[r] = sum;
            }
            for (std::size_t sub = 0; sub < block; ++sub)
            {
                amplitudes_[base | offsets[sub]] = result[sub];
            }
        }
    }

    void apply_jw_string(const cusv::Op &op)
    {
        const std::size_t control = std::size_t{1} << op.string_control;
        for (std::size_t index = 0; index < amplitudes_.size(); ++index)
        {
            if ((index & control) != 0u &&
                (__builtin_popcountll(static_cast<unsigned long long>(index & op.string_mask)) & 1))
            {
                amplitudes_[index] = -amplitudes_[index];
            }
        }
    }

    int n_ = 0;
    std::vector<std::complex<Real>> amplitudes_;
};

// One trajectory's measurement record: the outcome of every measured site, in
// the order the layers measure them.
using MeasurementRecord = std::vector<int>;

// Run one trajectory. With `forced` empty the outcomes are sampled from `rng`
// and appended to `record`; otherwise they are read from `forced` in order.
template <typename Real>
inline HostState<Real> run_trajectory(const std::vector<RppuLayer> &layers, int n,
                                      const MeasurementRecord *forced, MeasurementRecord &record,
                                      std::mt19937 &rng)
{
    HostState<Real> state(n);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::size_t consumed = 0;
    for (const RppuLayer &layer : layers)
    {
        const cusv::LayerOps ops = cusv::build_rppu_layer_ops(layer, n);
        for (const cusv::Op &op : ops.ops)
        {
            state.apply(op);
        }
        for (int site : ops.measure_sites)
        {
            int outcome = -1;
            if (forced != nullptr)
            {
                if (consumed >= forced->size())
                {
                    throw std::runtime_error("Replay record is shorter than the circuit.");
                }
                outcome = (*forced)[consumed];
            }
            const int drawn =
                state.measure(site, outcome, static_cast<Real>(uniform(rng)));
            if (forced == nullptr)
            {
                record.push_back(drawn);
            }
            ++consumed;
        }
    }
    if (forced != nullptr && consumed != forced->size())
    {
        throw std::runtime_error("Replay record is longer than the circuit.");
    }
    return state;
}

// The fermionic two-mode reduced density matrix, accumulated at the state's own
// precision.
//
// A separate, first-principles implementation rather than a call into
// ancilla_rdm.hpp, for one reason: that one accumulates into
// std::complex<double> whatever the input precision, which would throw away
// exactly the digits this replay exists to measure. The Jordan-Wigner sign is
// the inversion count of the permutation taking the ascending occupied-mode
// list to (retained, then environment), the same definition
// tests/rho_reduce_tests.cu pins the CUDA kernels against.
template <typename Real>
inline std::vector<std::complex<Real>> reduce_modes(const std::vector<std::complex<Real>> &psi,
                                                    int n, const std::vector<int> &kept,
                                                    bool fermionic)
{
    const int k = static_cast<int>(kept.size());
    const std::size_t dimension = std::size_t{1} << k;
    std::vector<std::complex<Real>> rho(dimension * dimension, std::complex<Real>(0, 0));
    std::size_t kept_mask = 0;
    for (int mode : kept)
    {
        kept_mask |= std::size_t{1} << mode;
    }
    auto sign_of = [&](std::size_t index) {
        if (!fermionic)
        {
            return Real(1);
        }
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
            if (std::find(kept.begin(), kept.end(), q) == kept.end() && ((index >> q) & 1u))
            {
                target.push_back(q);
            }
        }
        int inversions = 0;
        for (std::size_t a = 0; a < target.size(); ++a)
        {
            for (std::size_t b = a + 1; b < target.size(); ++b)
            {
                const auto pa = std::find(ascending.begin(), ascending.end(), target[a]);
                const auto pb = std::find(ascending.begin(), ascending.end(), target[b]);
                inversions += pa > pb ? 1 : 0;
            }
        }
        return (inversions & 1) ? Real(-1) : Real(1);
    };

    std::vector<std::complex<Real>> column(dimension);
    for (std::size_t env = 0; env < psi.size(); ++env)
    {
        if ((env & kept_mask) != 0u)
        {
            continue;
        }
        for (std::size_t r = 0; r < dimension; ++r)
        {
            std::size_t index = env;
            for (int slot = 0; slot < k; ++slot)
            {
                if ((r >> slot) & 1u)
                {
                    index |= std::size_t{1} << kept[static_cast<std::size_t>(slot)];
                }
            }
            column[r] = sign_of(index) * psi[index];
        }
        for (std::size_t r = 0; r < dimension; ++r)
        {
            for (std::size_t c = 0; c < dimension; ++c)
            {
                rho[r * dimension + c] += column[r] * std::conj(column[c]);
            }
        }
    }
    return rho;
}

// The quantities the request asks to compare, at one precision.
struct PairObservables
{
    long double re_g = 0.0L;
    long double im_g = 0.0L;
    long double re_f = 0.0L;
    long double im_f = 0.0L;
    long double g2 = 0.0L;
    long double f2 = 0.0L;
    long double parity_even = 0.0L;
    long double parity_odd = 0.0L;
    long double fn = 0.0L;
};

// G = rho[2][1] and F = -rho[3][0] in the |n_i n_j> basis indexed n_i + 2 n_j;
// the positions follow from particle number alone, so no Jordan-Wigner phase
// convention enters. See dist_metrics.hpp.
template <typename Real>
inline PairObservables pair_observables(const std::vector<std::complex<Real>> &rho)
{
    PairObservables out;
    const Real trace = (rho[0] + rho[5] + rho[10] + rho[15]).real();
    auto at = [&](int r, int c) { return rho[static_cast<std::size_t>(r * 4 + c)] / trace; };
    const std::complex<Real> g = at(2, 1);
    const std::complex<Real> f = -at(3, 0);
    out.re_g = static_cast<long double>(g.real());
    out.im_g = static_cast<long double>(g.imag());
    out.re_f = static_cast<long double>(f.real());
    out.im_f = static_cast<long double>(f.imag());
    out.g2 = out.re_g * out.re_g + out.im_g * out.im_g;
    out.f2 = out.re_f * out.re_f + out.im_f * out.im_f;
    out.parity_even = static_cast<long double>(at(0, 0).real() + at(3, 3).real());
    out.parity_odd = static_cast<long double>(at(1, 1).real() + at(2, 2).real());
    // The same cancellation-safe closed form the production path uses, carried
    // at long double so the comparison is between arithmetics rather than
    // between formulas.
    out.fn = 0.0L;
    if (out.g2 > 0.0L)
    {
        out.fn += 2.0L * out.g2 /
                  (std::sqrt(out.parity_even * out.parity_even + 4.0L * out.g2) + out.parity_even);
    }
    if (out.f2 > 0.0L)
    {
        out.fn += 2.0L * out.f2 /
                  (std::sqrt(out.parity_odd * out.parity_odd + 4.0L * out.f2) + out.parity_odd);
    }
    return out;
}

// One requested record.
struct ReplayRequest
{
    std::uint32_t realization = 0;
    int i = 0;
    int j = 0;
};

// `realization_id,pair_i,pair_j` per line; `#` starts a comment.
inline std::vector<ReplayRequest> read_replay_ids(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("Could not open replay id file " + path + ".");
    }
    std::vector<ReplayRequest> out;
    std::string line;
    std::size_t number = 0;
    while (std::getline(file, line))
    {
        ++number;
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos)
        {
            line.resize(hash);
        }
        if (line.find_first_not_of(" \t\r") == std::string::npos)
        {
            continue;
        }
        if (line.find("realization_id") != std::string::npos)
        {
            continue; // a header line
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        long realization = 0;
        int i = 0;
        int j = 0;
        if (!(stream >> realization >> i >> j))
        {
            throw std::runtime_error(path + " line " + std::to_string(number) +
                                     " is not `realization_id,pair_i,pair_j`.");
        }
        out.push_back({static_cast<std::uint32_t>(realization), i, j});
    }
    if (out.empty())
    {
        throw std::runtime_error(path + " names no records to replay.");
    }
    return out;
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

inline std::string replay_csv_header()
{
    return "run_id,N,periods,p,circ_type,master_seed,realization_id,pair_i,pair_j,separation,"
           "replay_format_version,high_precision_mantissa_bits,double_mantissa_bits,"
           "fp64_re_g,hp_re_g,fp64_im_g,hp_im_g,fp64_re_f,hp_re_f,fp64_im_f,hp_im_f,"
           "fp64_g2,hp_g2,fp64_f2,hp_f2,fp64_pair_fn,hp_pair_fn,"
           "delta_re_g,delta_im_g,delta_re_f,delta_im_f,delta_pair_fn,relative_delta_pair_fn,"
           "third_k,fp64_cut_i,hp_cut_i,fp64_cut_j,hp_cut_j,fp64_cut_k,hp_cut_k,"
           "fp64_min_cut,hp_min_cut,delta_min_cut,verdict\n";
}

// How a quantity moved between the two precisions.
//
//   structural_zero  -- exactly zero at both precisions. Not a small number; a
//                       zero, and no amount of precision will change it.
//   converged        -- the two arithmetics agree to double's own floor, so the
//                       value is determined by the physics: a real signal,
//                       however weak.
//   roundoff         -- a value at the double floor that moved. Nothing is
//                       there; the digits were arithmetic.
//   diverged         -- a value well above the floor that moved anyway. That
//                       cannot be roundoff, so the comparison itself is not
//                       clean -- the likeliest cause is that the two passes did
//                       not simulate the same trajectory, which is exactly what
//                       forcing the measurement record is supposed to prevent.
//                       Treat such a row as a bug report, not a measurement.
inline const char *replay_verdict(long double fp64_value, long double hp_value)
{
    const long double delta = std::fabs(fp64_value - hp_value);
    const long double scale = std::max(std::fabs(fp64_value), std::fabs(hp_value));
    if (scale <= 0.0L)
    {
        return "structural_zero";
    }
    if (delta / scale < 1.0e-10L)
    {
        return "converged";
    }
    // Below this, a value is the size of double's own error on a quantity of
    // order one, so a change of its own magnitude says only that it was never
    // there.
    if (scale < 1.0e-10L)
    {
        return "roundoff";
    }
    return "diverged";
}

struct ReplaySettings
{
    std::string ids_path;
    // Which third site to add when reporting the triple cuts. -1 picks the
    // site whose graph distance to the pair is smallest, which is not available
    // here, so the default is simply the lowest index that is not an endpoint.
    int third = -1;
};

// Replay every requested record and write the comparison CSV. Returns 0 on
// success.
inline int run_replay(const RunConfig &config, const ReplaySettings &settings, std::ostream &out)
{
    const std::vector<ReplayRequest> requests = read_replay_ids(settings.ids_path);
    out << "High-precision replay of " << requests.size() << " record(s) from "
        << settings.ids_path << '\n';
    out << "  N=" << config.n << ", periods=" << config.periods << ", p=" << config.p
        << ", master_seed=" << config.seed << '\n';
    out << "  double mantissa " << std::numeric_limits<double>::digits << " bits, high precision "
        << std::numeric_limits<long double>::digits << " bits ("
        << (std::numeric_limits<long double>::digits - std::numeric_limits<double>::digits)
        << " extra)\n";
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
    {
        out << "  WARNING: long double is not wider than double on this platform, so the "
               "comparison cannot separate roundoff from signal. Nothing below is meaningful.\n";
    }

    // Group by trajectory: one pair of simulations serves every request on it.
    std::vector<ReplayRequest> sorted = requests;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const ReplayRequest &a, const ReplayRequest &b) {
                         return a.realization < b.realization;
                     });

    std::string csv = replay_csv_header();
    char run_id[32];
    std::snprintf(run_id, sizeof(run_id), "s%016llx",
                  static_cast<unsigned long long>(config.seed));

    std::size_t index = 0;
    std::size_t trajectories = 0;
    while (index < sorted.size())
    {
        const std::uint32_t realization = sorted[index].realization;
        std::size_t end = index;
        while (end < sorted.size() && sorted[end].realization == realization)
        {
            ++end;
        }
        ++trajectories;

        // The circuit, regenerated exactly as the production loop drew it.
        std::vector<RppuLayer> layers;
        std::mt19937 layer_rng;
        {
            const std::uint64_t value = seeding::trajectory_seed(config.seed, realization);
            std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xffffffffu),
                                   static_cast<std::uint32_t>(value >> 32)};
            layer_rng.seed(sequence);
        }
        build_rppu_layers(layers, config.n, config.periods, config.p, true, layer_rng, false);

        // Pass 1 at double, sampling and recording the measurement record.
        MeasurementRecord record;
        std::mt19937 measure_rng;
        {
            const std::uint64_t value = seeding::splitmix64(
                seeding::trajectory_seed(config.seed, realization) ^ 0xa0761d0dull);
            std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xffffffffu),
                                   static_cast<std::uint32_t>(value >> 32)};
            measure_rng.seed(sequence);
        }
        const HostState<double> fp64 =
            run_trajectory<double>(layers, config.n, nullptr, record, measure_rng);

        // Pass 2 at long double, forcing that exact record.
        MeasurementRecord ignored;
        std::mt19937 unused;
        const HostState<long double> high =
            run_trajectory<long double>(layers, config.n, &record, ignored, unused);

        for (std::size_t r = index; r < end; ++r)
        {
            const ReplayRequest &request = sorted[r];
            const std::vector<int> pair{request.i, request.j};
            const PairObservables a =
                pair_observables(reduce_modes<double>(fp64.amplitudes(), config.n, pair, true));
            const PairObservables b = pair_observables(
                reduce_modes<long double>(high.amplitudes(), config.n, pair, true));

            int third = settings.third;
            if (third < 0 || third == request.i || third == request.j)
            {
                third = 0;
                while (third == request.i || third == request.j)
                {
                    ++third;
                }
            }
            std::vector<int> triple{request.i, request.j, third};
            std::sort(triple.begin(), triple.end());
            std::array<double, 3> cuts_a{};
            std::array<double, 3> cuts_b{};
            {
                const auto rho_a = reduce_modes<double>(fp64.amplitudes(), config.n, triple, true);
                const auto rho_b =
                    reduce_modes<long double>(high.amplitudes(), config.n, triple, true);
                std::array<double, 128> packed_a{};
                std::array<double, 128> packed_b{};
                double trace_a = 0.0;
                long double trace_b = 0.0L;
                for (int d = 0; d < 8; ++d)
                {
                    trace_a += rho_a[static_cast<std::size_t>(d * 8 + d)].real();
                    trace_b += rho_b[static_cast<std::size_t>(d * 8 + d)].real();
                }
                for (int rr = 0; rr < 8; ++rr)
                {
                    for (int cc = 0; cc < 8; ++cc)
                    {
                        const std::size_t slot = static_cast<std::size_t>(rr * 8 + cc);
                        packed_a[2u * slot] = rho_a[slot].real() / trace_a;
                        packed_a[2u * slot + 1u] = rho_a[slot].imag() / trace_a;
                        packed_b[2u * slot] =
                            static_cast<double>(rho_b[slot].real() / trace_b);
                        packed_b[2u * slot + 1u] =
                            static_cast<double>(rho_b[slot].imag() / trace_b);
                    }
                }
                analysis::cut_negativities<3>(packed_a.data(), true, cuts_a);
                analysis::cut_negativities<3>(packed_b.data(), true, cuts_b);
            }
            const double min_a = *std::min_element(cuts_a.begin(), cuts_a.end());
            const double min_b = *std::min_element(cuts_b.begin(), cuts_b.end());

            std::string line = run_id;
            auto number = [&](long double value) {
                line += ',';
                append_double(line, static_cast<double>(value));
            };
            auto integer = [&](long long value) { line += ',' + std::to_string(value); };
            integer(config.n);
            integer(config.periods);
            number(config.p);
            integer(static_cast<int>(config.type));
            line += ',' + std::to_string(config.seed);
            integer(realization);
            integer(request.i);
            integer(request.j);
            integer(util::periodic_distance(request.i, request.j, config.n));
            integer(REPLAY_FORMAT_VERSION);
            integer(std::numeric_limits<long double>::digits);
            integer(std::numeric_limits<double>::digits);
            number(a.re_g); number(b.re_g);
            number(a.im_g); number(b.im_g);
            number(a.re_f); number(b.re_f);
            number(a.im_f); number(b.im_f);
            number(a.g2); number(b.g2);
            number(a.f2); number(b.f2);
            number(a.fn); number(b.fn);
            number(std::fabs(a.re_g - b.re_g));
            number(std::fabs(a.im_g - b.im_g));
            number(std::fabs(a.re_f - b.re_f));
            number(std::fabs(a.im_f - b.im_f));
            number(std::fabs(a.fn - b.fn));
            number(std::max(std::fabs(a.fn), std::fabs(b.fn)) > 0.0L
                       ? std::fabs(a.fn - b.fn) / std::max(std::fabs(a.fn), std::fabs(b.fn))
                       : 0.0L);
            integer(third);
            for (int s = 0; s < 3; ++s)
            {
                number(cuts_a[static_cast<std::size_t>(s)]);
                number(cuts_b[static_cast<std::size_t>(s)]);
            }
            number(min_a);
            number(min_b);
            number(std::fabs(min_a - min_b));
            line += ',' + std::string(replay_verdict(a.fn, b.fn));
            line += '\n';
            csv += line;
        }
        index = end;
    }

    std::filesystem::path target(settings.ids_path);
    target.replace_extension();
    const std::string path = target.string() + "_replay.csv";
    std::ofstream file(path, std::ios::binary);
    file << csv;
    file.close();
    out << "  replayed " << trajectories << " trajectory/ies -> " << path << '\n';
    return 0;
}

} // namespace mipt::dist::replay

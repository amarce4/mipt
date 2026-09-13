#pragma once

// Offline re-analysis of a finished connected-zero triple run.
// ------------------------------------------------------------
//
// Changing a tolerance should not cost another circuit simulation. Everything
// the classification depends on is already on disk:
//
//   * the individual-outcome CSV carries, per (pair, third), the certified fGMN
//     interval, the three cut negativities, the minimum cut and the anchor
//     pair's own fN -- so both the fGMN positivity threshold and (downward) the
//     pair zero threshold can be re-applied without re-solving anything;
//   * the pair record binary carries, per (trajectory, pair), the occupation
//     data and |G|^2/|F|^2 that reconstruct the *exact* fermionic negativity at
//     any threshold through the closed form, together with the connectivity
//     flags -- so P(fN > eps | C) and the disconnected false-positive rate can
//     be swept over a broad interval;
//   * the RDM companion carries every unique triple's raw 8x8, so a selected
//     subset can be re-solved at a tighter interior-point tolerance.
//
// The audit reads all three and writes new files. **It never modifies its
// inputs** -- not the outcome CSV, not the aggregate, not the companion -- so
// it is safe to run against a live run's output directory.
//
// Why the pair-threshold sweep matters. The gap is P(C) - P(E_2): pairs the
// spacetime graph connects that carry no fermionic negativity. The obvious
// objection is that "no negativity" is an artefact of the threshold. Sweeping
// eps over many decades and finding the gap unchanged, while the *disconnected*
// pairs -- which are exact products and therefore measure the arithmetic floor
// alone -- stay flat at zero, rules that objection out operationally. Two
// curves is the whole point; a single P(fN > eps) curve cannot separate signal
// from floor.
//
// Direction matters for the pair threshold read off the outcome CSV. Anchors
// were selected at the run's own eps_2, so a *smaller* threshold selects a
// subset of rows that are all present, and a *larger* one would need pairs the
// run never analyzed. The sweep therefore only goes down from the recorded
// value, and says so rather than silently reporting a truncated set.

#include "mipt/dist_pair_gap.hpp"
#include "mipt/dist_records.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/env.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mipt::dist::gap::audit
{

inline constexpr int AUDIT_FORMAT_VERSION = 1;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

// A comma-separated list of positive reals, or the fallback when unset. An
// unparseable entry is refused rather than skipped: a silently dropped
// tolerance would leave a sweep with a hole in it and no indication.
inline std::vector<double> parse_tolerance_list(const char *name,
                                                const std::vector<double> &fallback)
{
    const std::string text = env::text(name, "");
    if (text.empty())
    {
        return fallback;
    }
    std::vector<double> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        const std::size_t begin = item.find_first_not_of(" \t");
        if (begin == std::string::npos)
        {
            continue;
        }
        const std::size_t end = item.find_last_not_of(" \t");
        const std::string trimmed = item.substr(begin, end - begin + 1);
        try
        {
            const double value = std::stod(trimmed);
            if (!(value > 0.0) || !std::isfinite(value))
            {
                throw std::invalid_argument("non-positive");
            }
            values.push_back(value);
        }
        catch (const std::exception &)
        {
            throw std::runtime_error(std::string(name) + " holds '" + trimmed +
                                     "', which is not a positive real number.");
        }
    }
    if (values.empty())
    {
        throw std::runtime_error(std::string(name) + " is set but names no tolerance.");
    }
    std::sort(values.begin(), values.end(), std::greater<double>());
    return values;
}

struct AuditSettings
{
    std::vector<double> pair_tols;
    std::vector<double> fgmn_tols;
    // Applied to every one-vs-rest cut when deciding "entangled across every
    // cut". Defaults to the value the run recorded.
    double cut_tol = 1.0e-10;
    // Only used when re-solving; recorded in the output metadata either way so
    // a sweep never claims a tolerance it did not use.
    std::string mosek_tol = "1e-8";
    bool resolve = false;
    std::string prefix; // output stem; defaults beside the outcome CSV
};

inline AuditSettings audit_settings_from_environment(double recorded_pair_tol,
                                                     double recorded_fgmn_tol,
                                                     double recorded_cut_tol)
{
    AuditSettings out;
    out.pair_tols = parse_tolerance_list("MIPT_DIST_AUDIT_PAIR_TOLS", {recorded_pair_tol});
    out.fgmn_tols = parse_tolerance_list("MIPT_DIST_AUDIT_FGMN_TOLS", {recorded_fgmn_tol});
    out.cut_tol = env::real("MIPT_DIST_AUDIT_CUT_TOL", recorded_cut_tol, 0.0, 1.0);
    out.mosek_tol = env::text("MIPT_DIST_AUDIT_MOSEK_TOL", "1e-8");
    out.resolve = env::boolean("MIPT_DIST_AUDIT_RESOLVE", false);
    out.prefix = env::text("MIPT_DIST_AUDIT_PREFIX", "");
    return out;
}

// ---------------------------------------------------------------------------
// What the outcome CSV says about the run that produced it
// ---------------------------------------------------------------------------

struct OutcomeMeta
{
    int n = 0;
    int periods = 0;
    double p = 0.0;
    int circ_type = 0;
    std::string circuit_name;
    std::string run_id;
    unsigned long long master_seed = 0;
    int statevector_precision = 0;
    int gap_format_version = 0;
    double pair_zero_tol = 0.0;
    double occupation_mi_tol = 0.0;
    double channel_floor = 0.0;
    double positive_tol = 0.0;
    double prefilter_tol = 0.0;
    double cut_positive_tol = 0.0;
    std::string mosek_tol;
};

// Read the header and first data row. Every provenance field is constant down
// the file, so one row settles the run's identity.
inline OutcomeMeta read_outcome_meta(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + path + " for reading.");
    }
    std::string line;
    if (!std::getline(file, line))
    {
        throw std::runtime_error(path + " is empty.");
    }
    std::string expected = outcome_csv_header();
    expected.pop_back();
    if (line != expected)
    {
        throw std::runtime_error(
            path + " has a different column layout than this binary writes. It was probably "
                   "produced by an older build; gap format version " +
            std::to_string(GAP_FORMAT_VERSION) +
            " added the certified fGMN interval, and an audit cannot reclassify rows that do "
            "not carry one.");
    }
    if (!std::getline(file, line))
    {
        throw std::runtime_error(path + " holds no outcome rows.");
    }
    const std::vector<std::string> fields = util::resume::split_csv_row(line);
    const auto &columns = outcome_columns();
    auto text = [&](const char *name) {
        const auto it = std::find(columns.begin(), columns.end(), name);
        if (it == columns.end() ||
            static_cast<std::size_t>(it - columns.begin()) >= fields.size())
        {
            throw std::runtime_error(path + " has no column '" + std::string(name) + "'.");
        }
        return fields[static_cast<std::size_t>(it - columns.begin())];
    };
    OutcomeMeta meta;
    meta.n = std::stoi(text("N"));
    meta.periods = std::stoi(text("periods"));
    meta.p = std::stod(text("p"));
    meta.circ_type = std::stoi(text("circ_type"));
    meta.circuit_name = text("circuit_name");
    meta.run_id = text("run_id");
    meta.master_seed = std::stoull(text("master_seed"));
    meta.statevector_precision = std::stoi(text("statevector_precision"));
    meta.gap_format_version = std::stoi(text("gap_format_version"));
    meta.pair_zero_tol = std::stod(text("pair_zero_tol"));
    meta.occupation_mi_tol = std::stod(text("occupation_mi_tol"));
    meta.channel_floor = std::stod(text("channel_floor"));
    meta.positive_tol = std::stod(text("fgmn_positive_tol"));
    meta.prefilter_tol = std::stod(text("prefilter_tol"));
    meta.cut_positive_tol = std::stod(text("cut_positive_tol"));
    meta.mosek_tol = text("mosek_tol");
    return meta;
}

// ---------------------------------------------------------------------------
// The fGMN threshold sweep
// ---------------------------------------------------------------------------

// One (pair threshold, fGMN threshold) cell of the sweep.
struct SweepCell
{
    std::uint64_t pairs = 0;
    std::uint64_t outcomes = 0;
    std::array<std::uint64_t, CERTIFIED_CLASS_COUNT> outcomes_by_class{};
    std::array<std::uint64_t, EXPLANATORY_CLASS_COUNT> pairs_by_explanation{};
    std::uint64_t pairs_with_positive_third = 0;
    std::uint64_t pairs_undetermined = 0;
};

// Indexed [pair_tol][fgmn_tol][role][class][separation].
class SweepTable
{
  public:
    SweepTable(std::size_t pair_tols, std::size_t fgmn_tols, int n)
        : pair_tols_(pair_tols), fgmn_tols_(fgmn_tols), max_separation_(n / 2)
    {
        cells_.assign(pair_tols * fgmn_tols * static_cast<std::size_t>(ROLE_COUNT) *
                          static_cast<std::size_t>(ZERO_CLASS_COUNT) *
                          static_cast<std::size_t>(max_separation_),
                      SweepCell{});
    }

    SweepCell &cell(std::size_t pt, std::size_t ft, int role, int zero_class, int separation)
    {
        const std::size_t index =
            ((((pt * fgmn_tols_ + ft) * static_cast<std::size_t>(ROLE_COUNT) +
               static_cast<std::size_t>(role)) *
                  static_cast<std::size_t>(ZERO_CLASS_COUNT) +
              static_cast<std::size_t>(zero_class)) *
             static_cast<std::size_t>(max_separation_)) +
            static_cast<std::size_t>(separation - 1);
        return cells_.at(index);
    }

    int max_separation() const { return max_separation_; }

  private:
    std::size_t pair_tols_ = 0;
    std::size_t fgmn_tols_ = 0;
    int max_separation_ = 0;
    std::vector<SweepCell> cells_;
};

struct SweepResult
{
    std::uint64_t rows = 0;
    std::uint64_t pairs = 0;
    std::uint64_t trajectories = 0;
    // Rows whose anchor fN exceeded the largest requested pair tolerance, i.e.
    // rows the sweep could not have produced from a run at that tolerance.
    std::uint64_t pairs_above_every_pair_tol = 0;
};

// One streaming pass over the outcome CSV, reclassifying every row at every
// requested (pair, fGMN) tolerance pair. A production outcome CSV is gigabytes,
// so it is never held in memory: only the current pair's rows are, which is
// N-2 of them.
//
// `primary` receives the pair summaries computed at pair_tols[0] / fgmn_tols[0]
// -- the regenerated summary file -- and `primary_aggregate` the matching
// aggregate.
inline SweepResult sweep_outcomes(const std::string &path, const AuditSettings &settings,
                                  const OutcomeMeta &meta, SweepTable &table,
                                  GapAggregate &primary_aggregate,
                                  const PairSummarySink &primary_sink)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + path + " for reading.");
    }
    std::string line;
    std::getline(file, line); // header, already validated by read_outcome_meta

    const auto &columns = outcome_columns();
    const OutcomeColumns c = OutcomeColumns::resolve();
    const std::size_t class_column = c.zero_class;

    SweepResult result;
    std::vector<RowSummary> group;
    std::vector<RowSummary> scratch;
    std::uint32_t last_realization = 0;
    bool seen = false;

    auto class_of = [&](const std::string &text) {
        for (int z = 0; z < ZERO_CLASS_COUNT; ++z)
        {
            if (text == zero_class_name(static_cast<ZeroClass>(z)))
            {
                return z;
            }
        }
        throw std::runtime_error(path + " names an unknown pair class '" + text + "'.");
    };

    auto flush = [&] {
        if (group.empty())
        {
            return;
        }
        ++result.pairs;
        const RowSummary &head = group.front();
        bool selected_anywhere = false;
        for (std::size_t pt = 0; pt < settings.pair_tols.size(); ++pt)
        {
            // An anchor belongs to this threshold's population only if its own
            // fermionic negativity is at or below it. Raising the threshold
            // past the run's own would need pairs that were never analyzed, so
            // it is a subset selection and nothing else.
            if (!(head.pair_fn <= settings.pair_tols[pt]))
            {
                continue;
            }
            selected_anywhere = true;
            for (std::size_t ft = 0; ft < settings.fgmn_tols.size(); ++ft)
            {
                const double threshold = settings.fgmn_tols[ft];
                scratch = group;
                for (RowSummary &row : scratch)
                {
                    row.certified = classify_certified(row.lower_bound, row.upper_bound,
                                                       threshold, !row.failed);
                    row.positive = row.certified == CertifiedClass::Positive;
                }
                const PairSummaryRow summary = summarize_pair(scratch.data(), scratch.size());
                SweepCell &cell =
                    table.cell(pt, ft, head.role, head.zero_class, head.separation);
                ++cell.pairs;
                cell.outcomes += scratch.size();
                cell.pairs_by_explanation[static_cast<std::size_t>(summary.explanation)] += 1u;
                for (const RowSummary &row : scratch)
                {
                    cell.outcomes_by_class[static_cast<std::size_t>(row.certified)] += 1u;
                }
                if (summary.thirds_positive > 0)
                {
                    ++cell.pairs_with_positive_third;
                }
                else if (summary.explanation == ExplanatoryClass::NumericallyUnresolved)
                {
                    ++cell.pairs_undetermined;
                }
                if (pt == 0 && ft == 0)
                {
                    primary_aggregate.absorb_pair(scratch.data(), scratch.size());
                    if (primary_sink)
                    {
                        primary_sink(summary);
                    }
                }
            }
        }
        if (!selected_anywhere)
        {
            ++result.pairs_above_every_pair_tol;
        }
        group.clear();
    };

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }
        const std::vector<std::string> fields = util::resume::split_csv_row(line);
        if (fields.size() != columns.size())
        {
            // A partial final line: the run was killed mid-write. Everything
            // before it is intact, and the audit reads rather than repairs.
            break;
        }
        RowSummary row = row_summary_from_fields(fields, c, path, settings.fgmn_tols.front(),
                                                 settings.cut_tol);
        row.zero_class = class_of(fields[class_column]);
        if (seen && row.realization < last_realization)
        {
            throw std::runtime_error(path + " is not in trajectory order.");
        }
        if (!seen || row.realization != last_realization)
        {
            ++result.trajectories;
        }
        last_realization = row.realization;
        seen = true;

        if (!group.empty() &&
            (group.front().realization != row.realization || group.front().role != row.role ||
             group.front().pair_i != row.pair_i || group.front().pair_j != row.pair_j))
        {
            flush();
        }
        group.push_back(row);
        ++result.rows;
    }
    flush();
    return result;
}

inline std::string sweep_csv(const AuditSettings &settings, const OutcomeMeta &meta,
                             SweepTable &table)
{
    std::string out =
        "run_id,N,circ_type,circuit_name,p,periods,master_seed,statevector_precision,"
        "audit_format_version,gap_format_version,recorded_pair_zero_tol,recorded_fgmn_positive_tol,"
        "pair_zero_tol,fgmn_positive_tol,cut_positive_tol,anchor_role,pair_class,separation,"
        "qualifying_pairs,outcomes,outcomes_positive,outcomes_bounded_below_threshold,"
        "outcomes_unresolved,outcomes_failed,pairs_with_positive_third,pairs_undetermined,"
        "r3_fraction";
    for (int e = 0; e < EXPLANATORY_CLASS_COUNT; ++e)
    {
        out += ",pairs_" + std::string(explanatory_class_name(static_cast<ExplanatoryClass>(e)));
    }
    out += '\n';

    for (std::size_t pt = 0; pt < settings.pair_tols.size(); ++pt)
    {
        for (std::size_t ft = 0; ft < settings.fgmn_tols.size(); ++ft)
        {
            for (int role = 0; role < ROLE_COUNT; ++role)
            {
                for (int z = 0; z < ZERO_CLASS_COUNT; ++z)
                {
                    for (int s = 1; s <= table.max_separation(); ++s)
                    {
                        const SweepCell &cell = table.cell(pt, ft, role, z, s);
                        if (cell.pairs == 0)
                        {
                            continue;
                        }
                        std::string line = meta.run_id;
                        field(line, static_cast<long long>(meta.n));
                        field(line, static_cast<long long>(meta.circ_type));
                        field(line, csv_quote(meta.circuit_name));
                        field(line, meta.p);
                        field(line, static_cast<long long>(meta.periods));
                        field(line, std::to_string(meta.master_seed));
                        field(line, static_cast<long long>(meta.statevector_precision));
                        field(line, static_cast<long long>(AUDIT_FORMAT_VERSION));
                        field(line, static_cast<long long>(meta.gap_format_version));
                        field(line, meta.pair_zero_tol);
                        field(line, meta.positive_tol);
                        field(line, settings.pair_tols[pt]);
                        field(line, settings.fgmn_tols[ft]);
                        field(line, settings.cut_tol);
                        field(line, std::string(role_name(static_cast<AnchorRole>(role))));
                        field(line, std::string(zero_class_name(static_cast<ZeroClass>(z))));
                        field(line, static_cast<long long>(s));
                        field(line, static_cast<long long>(cell.pairs));
                        field(line, static_cast<long long>(cell.outcomes));
                        for (int k : {0, 1, 2, 3})
                        {
                            field(line, static_cast<long long>(
                                            cell.outcomes_by_class[static_cast<std::size_t>(k)]));
                        }
                        field(line, static_cast<long long>(cell.pairs_with_positive_third));
                        field(line, static_cast<long long>(cell.pairs_undetermined));
                        field(line, positive_fraction(cell.pairs_with_positive_third, cell.pairs));
                        for (std::uint64_t count : cell.pairs_by_explanation)
                        {
                            field(line, static_cast<long long>(count));
                        }
                        line += '\n';
                        out += line;
                    }
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The pair-threshold sweep, from the main record binary
//
// This is the half the outcome CSV cannot supply: it covers *every* pair, not
// just the ones that qualified as connected zeros, so it gives both
// P(fN > eps | C) and the disconnected-control false-positive rate that says
// where the arithmetic floor is.
// ---------------------------------------------------------------------------

struct PairSweepCell
{
    std::uint64_t connected = 0;
    std::uint64_t connected_positive = 0;
    std::uint64_t disconnected = 0;
    std::uint64_t disconnected_positive = 0;
};

struct PairSweepResult
{
    bool available = false;
    std::string reason;
    std::uint64_t records = 0;
    int max_separation = 0;
    // [pair_tol][separation-1]
    std::vector<std::vector<PairSweepCell>> cells;
};

namespace detail
{

// One `key=value` line out of a record header.
inline std::string header_value(const std::string &header, const std::string &key)
{
    const std::string needle = key + "=";
    std::size_t pos = 0;
    while (pos < header.size())
    {
        const std::size_t end = header.find('\n', pos);
        const std::string line = header.substr(pos, end == std::string::npos ? end : end - pos);
        if (line.compare(0, needle.size(), needle) == 0)
        {
            return line.substr(needle.size());
        }
        if (end == std::string::npos)
        {
            break;
        }
        pos = end + 1;
    }
    return std::string();
}

inline std::vector<std::string> split_list(const std::string &text)
{
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        out.push_back(item);
    }
    return out;
}

// geometry_id -> separation, from the header's geometry table.
//
// The table is a delimited CSV block, exactly as append_geometry_table writes
// it: a `[geometry]` line, then the column names, then the rows, then `[end]`.
// The columns are looked up by name so a table that grows a column still reads.
inline std::map<int, int> geometry_separations(const std::string &header)
{
    std::map<int, int> out;
    const std::size_t begin = header.find("[geometry]\n");
    if (begin == std::string::npos)
    {
        return out;
    }
    std::size_t pos = begin + std::string("[geometry]\n").size();
    const std::size_t names_end = header.find('\n', pos);
    if (names_end == std::string::npos)
    {
        return out;
    }
    const std::vector<std::string> names = split_list(header.substr(pos, names_end - pos));
    std::size_t id_index = 0;
    std::size_t sep_index = 0;
    bool found_id = false;
    bool found_sep = false;
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if (names[i] == "geometry_id")
        {
            id_index = i;
            found_id = true;
        }
        if (names[i] == "separation")
        {
            sep_index = i;
            found_sep = true;
        }
    }
    // A k=3 table has l1/l2/l3 rather than one separation, and the pair sweep
    // is a k=2 question, so an unreadable table means "wrong file" not "parse
    // harder".
    if (!found_id || !found_sep)
    {
        return out;
    }
    pos = names_end + 1;
    while (pos < header.size())
    {
        const std::size_t end = header.find('\n', pos);
        if (end == std::string::npos)
        {
            break;
        }
        const std::string line = header.substr(pos, end - pos);
        if (line == "[end]")
        {
            break;
        }
        const std::vector<std::string> cells = split_list(line);
        if (cells.size() > std::max(id_index, sep_index))
        {
            try
            {
                out[std::stoi(cells[id_index])] = std::stoi(cells[sep_index]);
            }
            catch (const std::exception &)
            {
            }
        }
        pos = end + 1;
    }
    return out;
}

} // namespace detail

// Stream the record binary, recomputing the exact fermionic negativity of every
// pair record from its channel data and counting it against every threshold.
//
// The negativity is reconstructed with the same closed form the run used, so
// this is a re-reading of the recorded state, not an independent estimate --
// which is exactly what makes the comparison at two thresholds meaningful.
inline PairSweepResult sweep_pair_records(const std::string &path, const AuditSettings &settings)
{
    PairSweepResult result;
    records::FileInfo info;
    try
    {
        info = records::inspect(path);
    }
    catch (const std::exception &error)
    {
        result.reason = error.what();
        return result;
    }
    if (!info.exists)
    {
        result.reason = path + " does not exist.";
        return result;
    }
    if (info.version != records::FORMAT_VERSION)
    {
        result.reason = path + " is record format v" + std::to_string(info.version) +
                        ", which carries no connectivity flags; the pair sweep needs v2.";
        return result;
    }
    const std::vector<std::string> observables =
        detail::split_list(detail::header_value(info.header_text, "observables"));
    auto slot = [&](const char *name) -> long {
        const auto it = std::find(observables.begin(), observables.end(), name);
        return it == observables.end() ? -1L : static_cast<long>(it - observables.begin());
    };
    const long s_n_i = slot("n_i");
    const long s_n_j = slot("n_j");
    const long s_dnn = slot("dnn");
    const long s_g2 = slot("fg2");
    const long s_f2 = slot("ff2");
    if (s_n_i < 0 || s_n_j < 0 || s_dnn < 0 || s_g2 < 0 || s_f2 < 0)
    {
        result.reason =
            path + " carries observables [" + detail::header_value(info.header_text, "observables") +
            "], which cannot reconstruct the fermionic negativity. The sweep needs a "
            "parity-preserving run written with MIPT_DIST_RECORD_DETAIL=channel (the default) "
            "or full.";
        return result;
    }
    const std::map<int, int> separations = detail::geometry_separations(info.header_text);
    if (separations.empty())
    {
        result.reason = path + " has no readable geometry table.";
        return result;
    }
    int max_separation = 0;
    for (const auto &entry : separations)
    {
        max_separation = std::max(max_separation, entry.second);
    }
    if (max_separation <= 0)
    {
        result.reason = path + " names no positive separation.";
        return result;
    }

    const std::size_t values = (info.record_bytes - records::ID_BYTES) / sizeof(double);
    if (values != observables.size())
    {
        result.reason = path + " names " + std::to_string(observables.size()) +
                        " observables but carries " + std::to_string(values) + " per record.";
        return result;
    }

    result.cells.assign(settings.pair_tols.size(),
                        std::vector<PairSweepCell>(static_cast<std::size_t>(max_separation)));
    result.max_separation = max_separation;

    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(info.payload_offset));
    std::vector<char> buffer(info.record_bytes);
    const std::uint32_t connected_bit = static_cast<std::uint32_t>(FLAG_CONNECTED);
    const std::uint32_t evaluated_bit = static_cast<std::uint32_t>(FLAG_EVALUATED);
    for (std::uint64_t r = 0; r < info.record_count; ++r)
    {
        file.read(buffer.data(), static_cast<std::streamsize>(info.record_bytes));
        if (!file)
        {
            break;
        }
        std::uint16_t geometry = 0;
        std::uint32_t flags = 0;
        std::memcpy(&geometry, buffer.data() + 4, 2);
        std::memcpy(&flags, buffer.data() + 8, 4);
        if ((flags & evaluated_bit) == 0)
        {
            continue;
        }
        const auto found = separations.find(static_cast<int>(geometry));
        if (found == separations.end() || found->second < 1 || found->second > max_separation)
        {
            continue;
        }
        const double *payload = reinterpret_cast<const double *>(buffer.data() + records::ID_BYTES);
        const double n_i = payload[s_n_i];
        const double n_j = payload[s_n_j];
        const double dnn = payload[s_dnn];
        const double g2 = payload[s_g2];
        const double f2 = payload[s_f2];
        // p(11)=D, p(10)=n_i-D, p(01)=n_j-D, p(00)=1-n_i-n_j+D.
        const double parity_even = (1.0 - n_i - n_j + dnn) + dnn;
        const double parity_odd = (n_i - dnn) + (n_j - dnn);
        const double fn =
            parity_preserving_fermionic_negativity(parity_even, parity_odd, g2, f2);
        const bool connected = (flags & connected_bit) != 0;
        const std::size_t bin = static_cast<std::size_t>(found->second - 1);
        ++result.records;
        for (std::size_t pt = 0; pt < settings.pair_tols.size(); ++pt)
        {
            PairSweepCell &cell = result.cells[pt][bin];
            const bool positive = fn > settings.pair_tols[pt];
            if (connected)
            {
                ++cell.connected;
                cell.connected_positive += positive ? 1u : 0u;
            }
            else
            {
                ++cell.disconnected;
                cell.disconnected_positive += positive ? 1u : 0u;
            }
        }
    }
    result.available = true;
    return result;
}

inline std::string pair_sweep_csv(const AuditSettings &settings, const OutcomeMeta &meta,
                                  const PairSweepResult &sweep)
{
    std::string out = "run_id,N,circ_type,circuit_name,p,periods,statevector_precision,"
                      "audit_format_version,pair_zero_tol,separation,"
                      "connected_records,connected_positive,p_fn_positive_given_connected,"
                      "disconnected_records,disconnected_positive,"
                      "p_fn_positive_given_disconnected,gap\n";
    for (std::size_t pt = 0; pt < settings.pair_tols.size(); ++pt)
    {
        for (int s = 1; s <= sweep.max_separation; ++s)
        {
            const PairSweepCell &cell = sweep.cells[pt][static_cast<std::size_t>(s - 1)];
            if (cell.connected == 0 && cell.disconnected == 0)
            {
                continue;
            }
            std::string line = meta.run_id;
            field(line, static_cast<long long>(meta.n));
            field(line, static_cast<long long>(meta.circ_type));
            field(line, csv_quote(meta.circuit_name));
            field(line, meta.p);
            field(line, static_cast<long long>(meta.periods));
            field(line, static_cast<long long>(meta.statevector_precision));
            field(line, static_cast<long long>(AUDIT_FORMAT_VERSION));
            field(line, settings.pair_tols[pt]);
            field(line, static_cast<long long>(s));
            field(line, static_cast<long long>(cell.connected));
            field(line, static_cast<long long>(cell.connected_positive));
            const double connected_rate = positive_fraction(cell.connected_positive, cell.connected);
            field(line, connected_rate);
            field(line, static_cast<long long>(cell.disconnected));
            field(line, static_cast<long long>(cell.disconnected_positive));
            const double disconnected_rate =
                positive_fraction(cell.disconnected_positive, cell.disconnected);
            field(line, disconnected_rate);
            // P(C) - P(E_2 | C): the share of connected pairs the threshold
            // leaves unentangled. This is the gap, per separation, at this eps.
            field(line, cell.connected > 0 ? 1.0 - connected_rate
                                           : std::numeric_limits<double>::quiet_NaN());
            line += '\n';
            out += line;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Re-solving from the RDM companion
//
// The companion holds every unique triple's raw 8x8, so a triple whose interval
// straddled the threshold can be solved again at a tighter interior-point
// tolerance without re-simulating anything. That is the one operation the
// outcome CSV alone cannot do: reclassifying reuses the recorded interval,
// re-solving produces a new one.
// ---------------------------------------------------------------------------

struct ResolveResult
{
    bool available = false;
    std::string reason;
    std::uint64_t selected = 0;
    std::uint64_t solved = 0;
    std::uint64_t newly_resolved = 0;
    std::uint64_t failed = 0;
};

// Which triples are worth re-solving: the ones the recorded interval left
// unresolved, and the ones whose solve failed. A triple already certified
// either way gains nothing from a tighter solve.
inline std::set<std::pair<std::uint32_t, std::uint32_t>> unresolved_triples(
    const std::string &outcome_csv, double positive_tol, double cut_tol)
{
    std::set<std::pair<std::uint32_t, std::uint32_t>> out;
    std::ifstream file(outcome_csv, std::ios::binary);
    std::string line;
    std::getline(file, line);
    const auto &columns = outcome_columns();
    const OutcomeColumns c = OutcomeColumns::resolve();
    const std::size_t triple_column = static_cast<std::size_t>(
        std::find(columns.begin(), columns.end(), "triple_id") - columns.begin());
    const std::size_t first_use = c.first_use;
    while (std::getline(file, line))
    {
        const std::vector<std::string> fields = util::resume::split_csv_row(line);
        if (fields.size() != columns.size())
        {
            break;
        }
        if (fields[first_use] != "1")
        {
            continue; // one row owns each unique triple
        }
        const RowSummary row =
            row_summary_from_fields(fields, c, outcome_csv, positive_tol, cut_tol);
        if (row.certified == CertifiedClass::Unresolved || row.certified == CertifiedClass::Failed)
        {
            out.insert({row.realization,
                        static_cast<std::uint32_t>(std::stoul(fields[triple_column]))});
        }
    }
    return out;
}

inline ResolveResult resolve_from_companion(const std::string &outcome_csv,
                                            const std::string &rho3_path,
                                            const AuditSettings &settings, const OutcomeMeta &meta,
                                            const FgmnSolver &solver, const std::string &output,
                                            std::ostream &log)
{
    ResolveResult result;
    if (!solver)
    {
        result.reason = "no solver was supplied to the audit.";
        return result;
    }
    std::error_code error;
    if (rho3_path.empty() || !std::filesystem::exists(rho3_path, error))
    {
        result.reason = "the RDM companion " +
                        (rho3_path.empty() ? std::string("(not given)") : rho3_path) +
                        " is not there. Re-solving needs "
                        "MIPT_DIST_PAIR_GAP_STORE_RHO3=1 on the original run.";
        return result;
    }
    const auto wanted =
        unresolved_triples(outcome_csv, settings.fgmn_tols.front(), settings.cut_tol);
    result.selected = wanted.size();
    if (wanted.empty())
    {
        result.available = true;
        result.reason = "every triple is already certified at this threshold.";
        return result;
    }

    std::ifstream file(rho3_path, std::ios::binary);
    char preamble[RHO3_PREAMBLE];
    file.read(preamble, RHO3_PREAMBLE);
    if (!file || std::memcmp(preamble, RHO3_MAGIC, 8) != 0)
    {
        result.reason = rho3_path + " is not a pair-gap RDM companion.";
        return result;
    }
    std::uint32_t header_bytes = 0;
    std::memcpy(&header_bytes, preamble + 16, 4);
    file.seekg(static_cast<std::streamoff>(RHO3_PREAMBLE + header_bytes));

    std::string csv =
        "run_id,realization_id,triple_id,site_a,site_b,site_c,audit_mosek_tol,"
        "recorded_fgmn_positive_tol,fgmn_positive_tol,"
        "resolved_value,resolved_lower_bound,resolved_upper_bound,resolved_primal_residual,"
        "resolved_dual_residual,resolved_solver_gap,resolved_min_cut,resolved_class,"
        "resolved_status\n";

    std::vector<char> record(RHO3_RECORD_BYTES);
    while (file.read(record.data(), static_cast<std::streamsize>(RHO3_RECORD_BYTES)))
    {
        std::uint32_t realization = 0;
        std::uint32_t triple_id = 0;
        std::memcpy(&realization, record.data(), 4);
        std::memcpy(&triple_id, record.data() + 4, 4);
        if (wanted.find({realization, triple_id}) == wanted.end())
        {
            continue;
        }
        const double *raw = reinterpret_cast<const double *>(record.data() + 16);

        // Normalize and Hermitize exactly as evaluate_triple does, so the
        // solver sees the same matrix it saw the first time.
        const ancilla::SmallRdm rho = normalized_small_rdm(raw, 8);
        std::array<double, RHO3_DOUBLES> packed{};
        for (int r = 0; r < 8; ++r)
        {
            for (int c = 0; c < 8; ++c)
            {
                const std::complex<double> value = rho(r, c);
                packed[2u * static_cast<std::size_t>(r * 8 + c)] = value.real();
                packed[2u * static_cast<std::size_t>(r * 8 + c) + 1u] = value.imag();
            }
        }
        std::array<double, 3> cuts{};
        analysis::cut_negativities<3>(packed.data(), true, cuts);
        double min_cut = std::numeric_limits<double>::infinity();
        for (double cut : cuts)
        {
            min_cut = cut == cut ? std::min(min_cut, cut)
                                 : std::numeric_limits<double>::quiet_NaN();
        }

        const SolveOutcome outcome = solver(packed.data());
        ++result.solved;
        // Exactly the interval rule apply_solve uses, so an audited value and a
        // live one are the same computation. See apply_solve for why the gap
        // and the residuals enter as an uncertainty rather than as a bracket.
        const bool ok = outcome.status == FGMN_STATUS_OK && std::isfinite(outcome.value);
        double lower = std::numeric_limits<double>::quiet_NaN();
        double upper = std::numeric_limits<double>::quiet_NaN();
        if (ok)
        {
            double uncertainty = 0.0;
            for (double candidate :
                 {outcome.solver_gap, outcome.primal_residual, outcome.dual_residual})
            {
                if (std::isfinite(candidate))
                {
                    uncertainty = std::max(uncertainty, std::abs(candidate));
                }
            }
            lower = std::max(0.0, outcome.value - uncertainty);
            upper = std::max(0.0, outcome.value + uncertainty);
            if (min_cut == min_cut && min_cut < upper)
            {
                upper = min_cut;
            }
            if (lower > upper)
            {
                lower = std::numeric_limits<double>::quiet_NaN();
            }
        }
        else
        {
            ++result.failed;
        }
        const CertifiedClass verdict =
            classify_certified(lower, upper, settings.fgmn_tols.front(), ok);
        if (verdict != CertifiedClass::Unresolved && verdict != CertifiedClass::Failed)
        {
            ++result.newly_resolved;
        }

        std::string line = meta.run_id;
        field(line, static_cast<long long>(realization));
        field(line, static_cast<long long>(triple_id));
        for (int s = 0; s < 3; ++s)
        {
            field(line, static_cast<long long>(static_cast<unsigned char>(record[8 + s])));
        }
        field(line, csv_quote(settings.mosek_tol));
        field(line, meta.positive_tol);
        field(line, settings.fgmn_tols.front());
        field(line, outcome.value);
        field(line, lower);
        field(line, upper);
        field(line, outcome.primal_residual);
        field(line, outcome.dual_residual);
        field(line, outcome.solver_gap);
        field(line, min_cut);
        field(line, std::string(certified_class_name(verdict)));
        field(line, status_name(outcome.status));
        line += '\n';
        csv += line;
    }
    publish_atomically(output, csv);
    result.available = true;
    log << "  re-solved " << result.solved << " of " << result.selected
        << " unresolved triple(s) at mosek_tol=" << settings.mosek_tol << "; "
        << result.newly_resolved << " newly certified, " << result.failed << " failed -> "
        << output << '\n';
    return result;
}

// ---------------------------------------------------------------------------
// Driving the whole thing
// ---------------------------------------------------------------------------

struct AuditPaths
{
    std::string resolve;
    std::string fgmn_sweep;
    std::string pair_sweep;
    std::string pair_summary;
    std::string aggregate;
};

inline AuditPaths audit_paths(const std::string &outcome_csv, const std::string &prefix)
{
    std::string stem = prefix;
    if (stem.empty())
    {
        std::filesystem::path path(outcome_csv);
        path.replace_extension();
        stem = path.string() + "_audit";
    }
    AuditPaths out;
    out.resolve = stem + "_resolve.csv";
    out.fgmn_sweep = stem + "_fgmn_sweep.csv";
    out.pair_sweep = stem + "_pair_sweep.csv";
    out.pair_summary = stem + "_pair_summary.csv";
    out.aggregate = stem + "_aggregate.csv";
    return out;
}

// The whole audit. Returns 0 on success.
//
// `rho3_path` and `records_path` may be empty; the parts that need them are
// skipped with a printed reason rather than failing the run, because the fGMN
// sweep -- the part that answers the threshold question -- needs neither.
inline int run_audit(const std::string &outcome_csv, const std::string &rho3_path,
                     const std::string &records_path, std::ostream &out,
                     const FgmnSolver &solver = FgmnSolver())
{
    const OutcomeMeta meta = read_outcome_meta(outcome_csv);
    AuditSettings settings = audit_settings_from_environment(
        meta.pair_zero_tol, meta.positive_tol, meta.cut_positive_tol);

    out << "Gap audit of " << outcome_csv << '\n';
    out << "  run " << meta.run_id << ": N=" << meta.n << ", p=" << meta.p
        << ", circuit=" << meta.circuit_name << ", precision=fp" << meta.statevector_precision
        << ", gap format v" << meta.gap_format_version << '\n';
    out << "  recorded thresholds: pair_zero_tol=" << meta.pair_zero_tol
        << ", fgmn_positive_tol=" << meta.positive_tol << ", prefilter_tol=" << meta.prefilter_tol
        << ", cut_positive_tol=" << meta.cut_positive_tol << ", mosek_tol=" << meta.mosek_tol
        << '\n';

    // A pair threshold above the run's own would need pairs that were never
    // analyzed. Saying so is the difference between a truncated sweep and a
    // wrong one.
    std::vector<double> unreachable;
    for (double tol : settings.pair_tols)
    {
        if (tol > meta.pair_zero_tol)
        {
            unreachable.push_back(tol);
        }
    }
    if (!unreachable.empty())
    {
        out << "  NOTE: " << unreachable.size()
            << " requested pair tolerance(s) exceed the run's own " << meta.pair_zero_tol
            << ". Anchors were selected at that value, so a larger threshold cannot add the "
               "pairs it would admit; those columns of the fGMN sweep cover the recorded "
               "population only. The pair sweep below reads the record binary and is not "
               "limited this way.\n";
    }

    const AuditPaths paths = audit_paths(outcome_csv, settings.prefix);
    SweepTable table(settings.pair_tols.size(), settings.fgmn_tols.size(), meta.n);
    GapAggregate primary;
    primary.reset(meta.n);

    std::string summary = pair_summary_csv_header();
    RunConfig config;
    config.n = meta.n;
    config.periods = meta.periods;
    config.p = meta.p;
    config.type = static_cast<CircuitType>(meta.circ_type);
    config.seed = meta.master_seed;
    config.statevector_precision = meta.statevector_precision;
    config.pair_zero_tol = settings.pair_tols.front();
    config.occupation_mi_tol = meta.occupation_mi_tol;
    config.channel_floor = meta.channel_floor;
    config.pair_gap.positive_tol = settings.fgmn_tols.front();
    config.pair_gap.prefilter_tol = meta.prefilter_tol;
    config.pair_gap.cut_positive_tol = settings.cut_tol;
    config.pair_gap.mosek_tol_text = settings.resolve ? settings.mosek_tol : meta.mosek_tol;
    const RowPrefix prefix = row_prefix(config);

    const SweepResult swept =
        sweep_outcomes(outcome_csv, settings, meta, table, primary,
                       [&](const PairSummaryRow &row) {
                           append_pair_summary_row(summary, prefix, meta.run_id, row);
                       });

    out << "  read " << swept.rows << " outcome row(s) over " << swept.pairs << " pair(s) and "
        << swept.trajectories << " trajectory/ies.\n";
    if (swept.pairs_above_every_pair_tol > 0)
    {
        out << "  " << swept.pairs_above_every_pair_tol
            << " pair(s) had fN above every requested pair tolerance and are outside the swept "
               "population.\n";
    }

    publish_atomically(paths.fgmn_sweep, sweep_csv(settings, meta, table));
    publish_atomically(paths.pair_summary, summary);
    out << "  fGMN threshold sweep -> " << paths.fgmn_sweep << '\n';
    out << "  pair summary at pair_zero_tol=" << settings.pair_tols.front()
        << ", fgmn_positive_tol=" << settings.fgmn_tols.front() << " -> " << paths.pair_summary
        << '\n';

    if (!records_path.empty())
    {
        const PairSweepResult sweep = sweep_pair_records(records_path, settings);
        if (sweep.available)
        {
            publish_atomically(paths.pair_sweep, pair_sweep_csv(settings, meta, sweep));
            out << "  pair-threshold sweep over " << sweep.records << " record(s) -> "
                << paths.pair_sweep << '\n';
        }
        else
        {
            out << "  pair-threshold sweep skipped: " << sweep.reason << '\n';
        }
    }
    else
    {
        out << "  pair-threshold sweep skipped: no record binary given. Pass the run's "
               "*_records.bin as the third argument to get P(fN>eps|C) against the "
               "disconnected false-positive floor.\n";
    }

    if (settings.resolve)
    {
        const ResolveResult resolved = resolve_from_companion(
            outcome_csv, rho3_path, settings, meta, solver, paths.resolve, out);
        if (!resolved.available)
        {
            out << "  re-solve skipped: " << resolved.reason << '\n';
        }
        else if (resolved.solved == 0)
        {
            out << "  re-solve: " << resolved.reason << '\n';
        }
    }
    else if (!rho3_path.empty())
    {
        out << "  RDM companion " << rho3_path
            << " was named but not read: set MIPT_DIST_AUDIT_RESOLVE=1 to re-solve the "
               "unresolved triples at MIPT_DIST_AUDIT_MOSEK_TOL.\n";
    }

    out << "  inputs were not modified.\n";
    return 0;
}

} // namespace mipt::dist::gap::audit

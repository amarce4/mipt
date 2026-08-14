#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/free_energy_measure.hpp"
#include "mipt/entropy_resume.hpp"
#include "mipt/free_energy_resume.hpp"
#include "mipt/tmi/linalg.hpp"
#include "mipt/types.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void test_circuit_metadata()
{
    using namespace mipt;
    assert(parse_circuit_type(0) == CircuitType::MMS);
    assert(parse_circuit_type(4) == CircuitType::QubitRPPU);
    assert(uses_fermionic_trace(CircuitType::FermionRPPU));
    assert(uses_fermionic_trace(CircuitType::RFGS));
    assert(!uses_fermionic_trace(CircuitType::QubitRPPU));
    assert(requires_even_sites(CircuitType::Haar));
    assert(preserves_computational_parity(CircuitType::FermionRPPU));
    assert(preserves_computational_parity(CircuitType::RFGS));
    assert(preserves_computational_parity(CircuitType::QubitRPPU));
    assert(!preserves_computational_parity(CircuitType::MMS));
    assert(!preserves_computational_parity(CircuitType::Haar));

    bool rejected = false;
    try
    {
        (void)parse_circuit_type(5);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_linspace()
{
    const auto values = mipt::linspace(0.1, 0.5, 5);
    assert(values.size() == 5);
    assert(std::abs(values.front() - 0.1) < 1.0e-15);
    assert(std::abs(values[2] - 0.3) < 1.0e-15);
    assert(std::abs(values.back() - 0.5) < 1.0e-15);
}

void test_environment_parsing()
{
    constexpr const char *name = "MIPT_CORE_TEST_ENV";
    ::setenv(name, "YeS", 1);
    assert(mipt::env::boolean(name, false));
    ::setenv(name, "42", 1);
    assert(mipt::env::integer(name, 0, 0, 100) == 42);
    ::setenv(name, "2.5", 1);
    assert(std::abs(mipt::env::real(name, 0.0, 0.0, 3.0) - 2.5) < 1.0e-15);
    ::setenv(name, "outside", 1);
    assert(mipt::env::integer(name, 7, 0, 100) == 7);
    ::unsetenv(name);
}

void test_density_round_trip()
{
    using namespace mipt::io;
    const std::string path = "/tmp/mipt_core_density_test.bin";

    DensityFileMetadata metadata;
    metadata.spatial_dimension = 1;
    metadata.total_qubits = 4;
    metadata.kept_qubits = 2;
    metadata.matrix_dimension = 4;
    metadata.periods = 4;
    metadata.realizations = 1;
    metadata.resolution = 1;
    metadata.grid_x = 4;
    metadata.grid_y = 1;
    metadata.subsystem_count = 1;
    metadata.record_count = 1;
    metadata.p_min = 0.25;
    metadata.p_max = 0.25;
    metadata.subsystem_qubits = {1, 2};

    std::vector<double> payload(payload_value_count(metadata));
    for (std::size_t i = 0; i < payload.size(); ++i)
    {
        payload[i] = static_cast<double>(i) / 10.0;
    }

    {
        DensityMatrixWriter writer(path, metadata);
        writer.write_record(0, 3, 0.25, payload.data(), payload.size());
        writer.close();
    }

    DensityMatrixReader reader(path);
    assert(reader.metadata().subsystem_qubits == metadata.subsystem_qubits);
    DensityRecord record;
    assert(reader.read_record(record));
    assert(record.p_index == 0);
    assert(record.realization == 3);
    assert(std::abs(record.p - 0.25) < 1.0e-15);
    assert(record.rho_ri == payload);
    assert(!reader.read_record(record));
    std::remove(path.c_str());
}

void test_sequential_measurement_probabilities()
{
    // The unnormalized weights are [1,4,9,16]. Measuring q0=1 has
    // conditional probability 20/30. Given that result, measuring q1=0 has
    // conditional probability 4/20. Their product must equal the original
    // joint branch weight 4/30, and every collapse must restore unit norm.
    std::vector<std::complex<double>> state{
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};

    const auto first =
        mipt::free_energy::measure_collapse_host(state.data(), 2, 0, 0.5);
    assert(first.outcome == 1);
    assert(std::abs(first.conditional_probability - 2.0 / 3.0) < 1.0e-14);
    double norm = 0.0;
    for (const auto &value : state)
        norm += std::norm(value);
    assert(std::abs(norm - 1.0) < 1.0e-14);

    const auto second =
        mipt::free_energy::measure_collapse_host(state.data(), 2, 1, 0.9);
    assert(second.outcome == 0);
    assert(std::abs(second.conditional_probability - 0.2) < 1.0e-14);
    assert(std::abs(first.conditional_probability *
                        second.conditional_probability -
                    4.0 / 30.0) < 1.0e-14);
    norm = 0.0;
    for (const auto &value : state)
        norm += std::norm(value);
    assert(std::abs(norm - 1.0) < 1.0e-14);
    assert(std::abs(first.surprisal + second.surprisal +
                    std::log(4.0 / 30.0)) < 1.0e-14);
}
// --- free_energy.exe resume -------------------------------------------------
//
// The resume path reads free_energy.exe's own two CSVs, so everything below
// builds those files literally rather than mocking an interface.

namespace resume = mipt::free_energy::resume;

void test_resume_seed_stream_is_invertible()
{
    // Recovering the master seed from a samples row is what lets a resumed run
    // continue the same stream instead of re-drawing trajectories the
    // interrupted run already wrote.
    const std::uint64_t seeds[] = {
        0ULL, 1ULL, 42ULL, 0x9e3779b97f4a7c15ULL,
        0xdeadbeefcafef00dULL, ~0ULL, 13070133244918856633ULL};
    for (const std::uint64_t seed : seeds)
    {
        assert(resume::splitmix64_inverse(resume::splitmix64(seed)) == seed);
        assert(resume::splitmix64(resume::splitmix64_inverse(seed)) == seed);
    }

    // Newton's iteration must actually land on the modular inverse.
    for (const std::uint64_t odd : {0x94d049bb133111ebULL, 0xbf58476d1ce4e5b9ULL,
                                    3ULL, 0xffffffffffffffffULL})
    {
        assert(odd * resume::odd_inverse_mod_2_64(odd) == 1ULL);
    }

    const std::uint64_t master = 0x0123456789abcdefULL;
    for (const long trajectory : {0L, 1L, 21369L, 999999L})
    {
        const std::uint64_t drawn =
            resume::trajectory_seed_for(master, trajectory);
        assert(resume::recover_master_seed(drawn, trajectory) == master);
    }
}

void test_resume_csv_row_splitting()
{
    const auto plain = resume::split_csv_row("1,2,3");
    assert(plain.size() == 3 && plain[2] == "3");

    // circuit_name arrives quoted and contains spaces.
    const auto quoted = resume::split_csv_row(
        "18,0.335,2,\"Fermionic Random Parity Preserving Unitary\",1");
    assert(quoted.size() == 5);
    assert(quoted[3] == "Fermionic Random Parity Preserving Unitary");
    assert(quoted[4] == "1");

    // A comma inside quotes is not a separator, and "" is one literal quote.
    const auto tricky = resume::split_csv_row("a,\"b,c\"\"d\",e");
    assert(tricky.size() == 3);
    assert(tricky[1] == "b,c\"d");

    // An empty trailing field still counts.
    assert(resume::split_csv_row("a,").size() == 2);
}

void test_resume_half_entropy_schedule()
{
    // Disabled: the app passes entropy_max_t = -1 in that case.
    assert(!resume::records_half_entropy(0, false, -1, 1));
    assert(!resume::records_half_entropy(5, false, -1, 1));

    // Default schedule: every timestep through 2L, plus the initial state.
    assert(resume::records_half_entropy(0, true, 36, 1));
    assert(resume::records_half_entropy(36, true, 36, 1));
    assert(!resume::records_half_entropy(37, true, 36, 1));

    // A stride still samples the endpoint, which is how the writer's
    // `t == entropy_max_t` clause behaves.
    assert(resume::records_half_entropy(10, true, 25, 5));
    assert(!resume::records_half_entropy(11, true, 25, 5));
    assert(resume::records_half_entropy(25, true, 25, 5));
}

void test_resume_running_stats_round_trip()
{
    // The time-series CSV stores mean and sample stddev, never m2. Resuming
    // reconstructs the Welford triple from those, so an interrupted-then-
    // resumed accumulation must match one that never stopped.
    std::vector<double> first;
    std::vector<double> second;
    for (int i = 0; i < 400; ++i)
    {
        first.push_back(3.5 + 0.01 * i + 0.7 * std::sin(0.3 * i));
    }
    for (int i = 0; i < 250; ++i)
    {
        second.push_back(-1.25 + 0.02 * i + std::cos(0.11 * i));
    }

    mipt::util::RunningStats uninterrupted;
    mipt::util::RunningStats before_break;
    for (const double value : first)
    {
        uninterrupted.add(value);
        before_break.add(value);
    }

    // Round-trip through the writer's 17-digit text form.
    std::ostringstream text;
    text << std::setprecision(17) << before_break.mean << ','
         << before_break.sample_stddev();
    const auto fields = resume::split_csv_row(text.str());
    auto restored = resume::rebuild_stats(
        std::stod(fields[0]), std::stod(fields[1]), before_break.count);

    for (const double value : second)
    {
        uninterrupted.add(value);
        restored.add(value);
    }

    assert(restored.count == uninterrupted.count);
    assert(std::abs(restored.mean - uninterrupted.mean) <
           1.0e-12 * std::abs(uninterrupted.mean));
    assert(std::abs(restored.sample_stddev() - uninterrupted.sample_stddev()) <
           1.0e-10 * uninterrupted.sample_stddev());

    // An empty accumulator is written as nan and must come back empty, not as
    // a zero-valued sample.
    assert(resume::rebuild_stats(
               std::nan(""), std::nan(""), 17).count == 0);
    assert(resume::rebuild_stats(1.0, 0.0, 0).count == 0);
    // A single sample has no sample stddev; m2 must be 0, not 0 * -1.
    const auto single = resume::rebuild_stats(2.5, 0.0, 1);
    assert(single.count == 1 && single.m2 == 0.0);
}

// Writes a samples/time-series pair the way free_energy.exe would after
// `completed` trajectories, and returns the samples text so a test can corrupt
// its tail.
std::string make_samples_csv(int n, double p, long completed, std::uint64_t master)
{
    std::ostringstream csv;
    csv << std::setprecision(17) << resume::kSamplesHeader << '\n';
    for (long trajectory = 0; trajectory < completed; ++trajectory)
    {
        csv << n << ',' << p << ",2,\"Fermionic Random Parity Preserving "
               "Unitary\",1," << trajectory << ",1,"
            << resume::trajectory_seed_for(master, trajectory)
            << ",72,432,504,3050,2606,1729.5,1479.7,3.42,3.42,0.19\n";
    }
    return csv.str();
}

std::string make_timeseries_csv(
    int n, double p, long requested, long completed, int total_timesteps,
    int entropy_max_t)
{
    std::ostringstream csv;
    csv << std::setprecision(17) << resume::kTimeseriesHeader << '\n';
    for (int t = 0; t <= total_timesteps; ++t)
    {
        const bool entropy = resume::records_half_entropy(t, true, entropy_max_t, 1);
        csv << n << ',' << p << ",2,\"Fermionic Random Parity Preserving "
               "Unitary\",1," << requested << ',' << completed << ',' << t << ','
            << static_cast<double>(t) / n << ",production,"
            << 4.0 * t << ',' << 1.5 << ',' << 0.01 << ',';
        if (t == 0)
        {
            csv << "nan,nan,nan,";
        }
        else
        {
            csv << 0.19 << ',' << 0.003 << ',' << 2.2e-5 << ',';
        }
        if (entropy)
        {
            csv << 8.27 << ',' << 0.002 << ',' << 1.3e-5 << ',' << completed;
        }
        else
        {
            csv << "nan,nan,nan,0";
        }
        csv << '\n';
    }
    return csv.str();
}

void write_text(const std::filesystem::path &path, const std::string &text)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
    out << text;
    out.close();
    assert(std::filesystem::exists(path));
}

void test_resume_checkpoint_scanning()
{
    const auto directory =
        std::filesystem::temp_directory_path() / "mipt_resume_core_tests";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    constexpr int n = 18;
    constexpr double p = 0.335;
    constexpr int total_timesteps = 504;
    constexpr int entropy_max_t = 36;
    constexpr long requested = 25000;
    constexpr long completed = 21370;
    constexpr std::uint64_t master = 0xfeedfacecafebeefULL;

    const auto samples_path = directory / "samples.csv";
    const auto timeseries_path = directory / "timeseries.csv";
    write_text(samples_path, make_samples_csv(n, p, completed, master));
    write_text(
        timeseries_path,
        make_timeseries_csv(
            n, p, requested, completed, total_timesteps, entropy_max_t));

    resume::RunIdentity identity;
    identity.n = n;
    identity.p = p;
    identity.circ_type = 2;
    identity.init_state = 1;
    identity.requested_realizations = requested;

    auto scan = resume::scan_samples(
        samples_path, resume::kSamplesHeader, identity);
    assert(scan.completed == completed);
    assert(scan.next_trajectory == completed);
    assert(scan.has_master_seed && scan.master_seed == master);
    assert(!resume::has_partial_tail(scan));

    const auto checkpoint = resume::load_timeseries(
        timeseries_path,
        resume::kTimeseriesHeader,
        identity,
        total_timesteps,
        true,
        entropy_max_t,
        1);
    assert(checkpoint.completed == completed);
    assert(checkpoint.stats.size() ==
           static_cast<std::size_t>(total_timesteps) + 1);
    assert(checkpoint.stats[0].cumulative_f.count ==
           static_cast<std::uint64_t>(completed));
    // t=0 has no F/tL sample; it is written as nan and must not become a zero.
    assert(checkpoint.stats[0].f_over_tl.count == 0);
    assert(checkpoint.stats[0].half_entropy.count ==
           static_cast<std::uint64_t>(completed));
    assert(checkpoint.stats[entropy_max_t].half_entropy.count ==
           static_cast<std::uint64_t>(completed));
    assert(checkpoint.stats[entropy_max_t + 1].half_entropy.count == 0);
    assert(checkpoint.stats[total_timesteps].f_over_tl.count ==
           static_cast<std::uint64_t>(completed));

    // A kill mid-flush leaves an unterminated final row. It must be dropped,
    // and every complete row before it kept.
    auto torn = make_samples_csv(n, p, completed, master);
    torn += "18,0.335,2,\"Fermionic Random Parity Prese";
    write_text(samples_path, torn);
    scan = resume::scan_samples(samples_path, resume::kSamplesHeader, identity);
    assert(scan.completed == completed);
    assert(resume::has_partial_tail(scan));
    resume::trim_partial_tail(samples_path, scan);
    assert(std::filesystem::file_size(samples_path) == scan.valid_bytes);
    const auto retrimmed =
        resume::scan_samples(samples_path, resume::kSamplesHeader, identity);
    assert(retrimmed.completed == completed);
    assert(!resume::has_partial_tail(retrimmed));

    const auto refuses = [](auto &&action) {
        bool threw = false;
        try
        {
            action();
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        assert(threw);
    };

    // A different N, p, circuit or init_state is a different run that happens
    // to share a filename; resuming into it would pool unrelated trajectories.
    refuses([&]() {
        auto other = identity;
        other.p = 0.17;
        (void)resume::scan_samples(
            samples_path, resume::kSamplesHeader, other);
    });
    refuses([&]() {
        auto other = identity;
        other.n = 16;
        (void)resume::load_timeseries(
            timeseries_path, resume::kTimeseriesHeader, other,
            total_timesteps, true, entropy_max_t, 1);
    });
    // A time series from a shorter circuit has the wrong number of rows.
    refuses([&]() {
        (void)resume::load_timeseries(
            timeseries_path, resume::kTimeseriesHeader, identity,
            total_timesteps - 2, true, entropy_max_t, 1);
    });
    // Resuming under a different entropy schedule would silently produce a
    // column whose per-t sample counts disagree.
    refuses([&]() {
        (void)resume::load_timeseries(
            timeseries_path, resume::kTimeseriesHeader, identity,
            total_timesteps, true, entropy_max_t + 4, 1);
    });
    refuses([&]() {
        (void)resume::load_timeseries(
            timeseries_path, resume::kTimeseriesHeader, identity,
            total_timesteps, false, -1, 1);
    });
    // An older schema must not be appended to.
    refuses([&]() {
        (void)resume::scan_samples(
            samples_path, "N,p,circ_type,F_total", identity);
    });
    // Two runs concatenated leave a trajectory index that does not follow.
    refuses([&]() {
        auto doubled = make_samples_csv(n, p, 4, master);
        doubled += make_samples_csv(n, p, 4, master).substr(
            resume::kSamplesHeader.size() + 1);
        write_text(samples_path, doubled);
        (void)resume::scan_samples(
            samples_path, resume::kSamplesHeader, identity);
    });

    // A header-only file has nothing to resume and must not look torn.
    write_text(samples_path, std::string(resume::kSamplesHeader) + "\n");
    const auto empty =
        resume::scan_samples(samples_path, resume::kSamplesHeader, identity);
    assert(empty.completed == 0 && !resume::has_partial_tail(empty));

    std::filesystem::remove_all(directory);
}

// --- entropy.exe resume -----------------------------------------------------
//
// entropy.exe's aggregated CSV is its only output and its whole checkpoint, so
// these tests render it exactly as write_results does and read it back.

namespace entropy_resume = mipt::entropy::resume;

// The projection write_results performs, minus the CUDA-Q dependency that
// pulling in entropy.hpp would add. Kept in step with it by the shared
// kResultsHeader constant, which is what a schema drift would break first.
std::string render_entropy_csv(
    int n,
    int periods,
    double p,
    int requested_realizations,
    int circ_type,
    const std::vector<mipt::util::RunningStats> &s1,
    const std::vector<mipt::util::RunningStats> &s2)
{
    std::ostringstream csv;
    csv << std::setprecision(17) << entropy_resume::kResultsHeader << '\n';
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int max_kept = n / 2;
    for (int kept = 2; kept <= max_kept; ++kept)
    {
        const double x = (static_cast<double>(n) / pi) *
                         std::sin(pi * static_cast<double>(kept) / static_cast<double>(n));
        const auto &a = s1[static_cast<std::size_t>(kept)];
        const auto &b = s2[static_cast<std::size_t>(kept)];
        const bool has_s1 = a.count > 0;
        const double nan_value = std::numeric_limits<double>::quiet_NaN();
        csv << n << ',' << periods << ',' << p << ',' << requested_realizations << ','
            << b.count << ',' << circ_type << ',' << "\"Test Circuit\"" << ',' << 0 << ','
            << kept << ',' << x << ',' << std::log(x) << ','
            << (has_s1 ? a.mean : nan_value) << ','
            << (has_s1 ? a.sample_stddev() : nan_value) << ','
            << (has_s1 ? a.standard_error() : nan_value) << ','
            << b.mean << ',' << b.sample_stddev() << ',' << b.standard_error() << ','
            << n << ',' << static_cast<std::uint64_t>(n) * b.count << '\n';
    }
    return csv.str();
}

void test_entropy_resume()
{
    const auto directory =
        std::filesystem::temp_directory_path() / "mipt_entropy_resume_tests";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::string path = (directory / "scan.csv").string();

    constexpr int n = 16;
    constexpr int periods = 12;
    constexpr double p = 0.25;
    constexpr int requested = 40;
    constexpr int circ_type = 1;
    const int max_kept = n / 2;
    const int max_s1_block = 5;  // S1 capped below N/2, so the tail is NaN

    std::vector<mipt::util::RunningStats> s1(static_cast<std::size_t>(n + 1));
    std::vector<mipt::util::RunningStats> s2(static_cast<std::size_t>(n + 1));
    std::vector<mipt::util::RunningStats> reference_s1 = s1;
    std::vector<mipt::util::RunningStats> reference_s2 = s2;

    const auto add_trajectory = [&](std::vector<mipt::util::RunningStats> &a,
                                    std::vector<mipt::util::RunningStats> &b,
                                    int trajectory) {
        for (int kept = 2; kept <= max_kept; ++kept)
        {
            const double value =
                0.3 * kept + 0.05 * std::sin(0.7 * trajectory + kept);
            b[static_cast<std::size_t>(kept)].add(value);
            if (kept <= max_s1_block)
            {
                a[static_cast<std::size_t>(kept)].add(value * 1.4 + 0.01 * trajectory);
            }
        }
    };

    constexpr int before_break = 17;
    for (int t = 0; t < before_break; ++t)
    {
        add_trajectory(s1, s2, t);
        add_trajectory(reference_s1, reference_s2, t);
    }
    {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        out << render_entropy_csv(n, periods, p, requested, circ_type, s1, s2);
    }

    entropy_resume::RunIdentity identity;
    identity.n = n;
    identity.periods = periods;
    identity.p = p;
    identity.requested_realizations = requested;
    identity.circ_type = circ_type;

    auto checkpoint = entropy_resume::load(path, identity, max_s1_block);
    assert(checkpoint.completed == before_break);
    for (int kept = 2; kept <= max_kept; ++kept)
    {
        const auto &restored = checkpoint.s2[static_cast<std::size_t>(kept)];
        const auto &expected = s2[static_cast<std::size_t>(kept)];
        assert(restored.count == expected.count);
        assert(std::abs(restored.mean - expected.mean) < 1.0e-12);
        assert(std::abs(restored.sample_stddev() - expected.sample_stddev()) <
               1.0e-10 * std::max(1.0e-300, expected.sample_stddev()));

        // A block whose S1 was skipped must come back empty, not as a zero
        // sample -- the CSV writes NaN there precisely to keep them distinct.
        const auto &restored_s1 = checkpoint.s1[static_cast<std::size_t>(kept)];
        if (kept <= max_s1_block)
        {
            assert(restored_s1.count == static_cast<std::uint64_t>(before_break));
        }
        else
        {
            assert(restored_s1.count == 0);
            assert(restored_s1.mean == 0.0);
        }
    }

    // Continuing from the checkpoint must land where an uninterrupted run would.
    for (int t = before_break; t < requested; ++t)
    {
        add_trajectory(checkpoint.s1, checkpoint.s2, t);
        add_trajectory(reference_s1, reference_s2, t);
    }
    for (int kept = 2; kept <= max_kept; ++kept)
    {
        const auto &got2 = checkpoint.s2[static_cast<std::size_t>(kept)];
        const auto &want2 = reference_s2[static_cast<std::size_t>(kept)];
        assert(got2.count == want2.count);
        assert(std::abs(got2.mean - want2.mean) < 1.0e-12);
        assert(std::abs(got2.sample_stddev() - want2.sample_stddev()) <
               1.0e-10 * std::max(1.0e-300, want2.sample_stddev()));
        const auto &got1 = checkpoint.s1[static_cast<std::size_t>(kept)];
        const auto &want1 = reference_s1[static_cast<std::size_t>(kept)];
        assert(got1.count == want1.count);
        assert(std::abs(got1.mean - want1.mean) < 1.0e-12);
    }

    const auto refuses = [](auto &&action) {
        bool threw = false;
        try
        {
            action();
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        assert(threw);
    };

    // A different run that happens to share the output path.
    refuses([&]() {
        auto other = identity;
        other.p = 0.3;
        (void)entropy_resume::load(path, other, max_s1_block);
    });
    refuses([&]() {
        auto other = identity;
        other.periods = 13;
        (void)entropy_resume::load(path, other, max_s1_block);
    });
    refuses([&]() {
        auto other = identity;
        other.requested_realizations = 41;
        (void)entropy_resume::load(path, other, max_s1_block);
    });
    refuses([&]() {
        auto other = identity;
        other.n = 14;  // wrong number of block rows
        (void)entropy_resume::load(path, other, max_s1_block);
    });
    // Resuming under a different S1 cap would leave the S1 columns averaged
    // over a different number of trajectories than the S2 ones.
    refuses([&]() { (void)entropy_resume::load(path, identity, max_s1_block + 1); });
    refuses([&]() { (void)entropy_resume::load(path, identity, 0); });

    // An S2-only file from before 2026-08-05 resumes as an S2-only scan, and is
    // refused when the run would compute S1.
    std::string legacy(entropy_resume::kLegacyS2OnlyHeader);
    legacy += '\n';
    for (int kept = 2; kept <= max_kept; ++kept)
    {
        const auto &b = s2[static_cast<std::size_t>(kept)];
        std::ostringstream row;
        row << std::setprecision(17) << n << ',' << periods << ',' << p << ','
            << requested << ',' << b.count << ',' << circ_type << ",\"Test Circuit\",0,"
            << kept << ",1,0," << b.mean << ',' << b.sample_stddev() << ','
            << b.standard_error() << ',' << n << ','
            << static_cast<std::uint64_t>(n) * b.count << '\n';
        legacy += row.str();
    }
    const std::string legacy_path = (directory / "legacy.csv").string();
    {
        std::ofstream out(legacy_path, std::ios::out | std::ios::trunc);
        out << legacy;
    }
    refuses([&]() { (void)entropy_resume::load(legacy_path, identity, max_s1_block); });
    const auto legacy_checkpoint = entropy_resume::load(legacy_path, identity, 0);
    assert(legacy_checkpoint.completed == before_break);
    assert(legacy_checkpoint.s2[2].count == static_cast<std::uint64_t>(before_break));
    assert(legacy_checkpoint.s1[2].count == 0);

    // stderr = stddev / sqrt(count) ties the two spread columns to the count,
    // so an edited or concatenated file gives itself away for free. Here the
    // count is doubled while the spreads are left alone.
    refuses([&]() {
        // Rewrite only the completed_realizations field (index 4), leaving the
        // spreads as they were -- a hand edit, not a re-render.
        const std::string original =
            render_entropy_csv(n, periods, p, requested, circ_type, s1, s2);
        std::string edited;
        std::size_t cursor = 0;
        bool first_line = true;
        while (cursor < original.size())
        {
            const std::size_t newline = original.find('\n', cursor);
            const std::string line = original.substr(cursor, newline - cursor);
            cursor = newline + 1;
            if (first_line)
            {
                first_line = false;
                edited += line + '\n';
                continue;
            }
            auto fields = mipt::util::resume::split_csv_row(line);
            fields[4] = std::to_string(before_break * 2);
            for (std::size_t i = 0; i < fields.size(); ++i)
            {
                edited += (i == 6 ? '"' + fields[i] + '"' : fields[i]);
                edited += (i + 1 == fields.size()) ? '\n' : ',';
            }
        }
        const std::string edited_path = (directory / "edited.csv").string();
        std::ofstream(edited_path, std::ios::out | std::ios::trunc) << edited;
        (void)entropy_resume::load(edited_path, identity, max_s1_block);
    });

    std::filesystem::remove_all(directory);
}
// The Renyi-2 branch of the TMI entropy primitives.
//
// These are the only pieces sim_tmi.exe's order-2 mode uses that are not
// behind CUDA, so pinning them here keeps them covered on a machine with no
// GPU.  The device purity reduction has its own check in
// tests/rdm_gram_tests.cu.
void test_tmi_renyi2_primitives()
{
    using namespace mipt::tmi;
    const double tol = 1.0e-12;

    // A maximally mixed rho_A of dimension d has purity 1/d, so S_2 = log2 d --
    // the same as its von Neumann entropy, which is what makes this a usable
    // cross-check on both branches at once.
    for (std::size_t dim : {std::size_t{2}, std::size_t{4}, std::size_t{8}})
    {
        std::vector<C64> rho(dim * dim, C64(0.0, 0.0));
        for (std::size_t i = 0; i < dim; ++i)
        {
            rho[i * dim + i] = C64(1.0 / static_cast<double>(dim), 0.0);
        }
        EntropyWorkspace ws;
        std::vector<C64> copy = rho;
        const double s2 = entropy_hermitian_inplace(copy, dim, ws, EntropyOrder::Renyi2);
        copy = rho;
        const double s1 = entropy_hermitian_inplace(copy, dim, ws, EntropyOrder::VonNeumann);
        assert(std::abs(s2 - std::log2(static_cast<double>(dim))) < tol);
        assert(std::abs(s1 - s2) < 1.0e-9);
    }

    // A pure rho has purity 1 and must give exactly 0, not -log2(1+eps).
    {
        std::vector<C64> pure(4, C64(0.0, 0.0));
        pure[0] = C64(1.0, 0.0);
        EntropyWorkspace ws;
        assert(entropy_hermitian_inplace(pure, 2, ws, EntropyOrder::Renyi2) == 0.0);
        assert(renyi2_from_purity(1.0 + 1.0e-9) == 0.0);
    }

    // A rank-2 state with weights (q, 1-q): purity q^2 + (1-q)^2.
    {
        const double q = 0.3;
        std::vector<C64> rho(4, C64(0.0, 0.0));
        rho[0] = C64(q, 0.0);
        rho[3] = C64(1.0 - q, 0.0);
        EntropyWorkspace ws;
        const double expected = -std::log2(q * q + (1.0 - q) * (1.0 - q));
        assert(std::abs(entropy_hermitian_inplace(rho, 2, ws, EntropyOrder::Renyi2) -
                        expected) < tol);

        // Same state seen as Schmidt coefficients: purity is the fourth moment.
        const std::vector<double> sigma{std::sqrt(q), std::sqrt(1.0 - q)};
        assert(std::abs(renyi2_from_singular_values_raw(sigma) - expected) < tol);
        assert(std::abs(entropy_from_singular_values_ordered(sigma, EntropyOrder::Renyi2) -
                        expected) < tol);
    }

    // Purity must ignore storage order and be blind to a non-Hermitian
    // perturbation only after hermitizing -- the sum of |a_ij|^2 is Tr(a^2)
    // only for Hermitian a, which is why the order-2 branch hermitizes first.
    {
        std::vector<C64> rho{C64(0.6, 0.0), C64(0.2, 0.1), C64(0.2, -0.1), C64(0.4, 0.0)};
        const double direct = purity_hermitian(rho.data(), 2);
        const double expected = 0.6 * 0.6 + 0.4 * 0.4 + 2.0 * (0.2 * 0.2 + 0.1 * 0.1);
        assert(std::abs(direct - expected) < tol);
    }

    // Order 1 is the default on every overload that gained the parameter.
    {
        std::vector<C64> rho(4, C64(0.0, 0.0));
        rho[0] = C64(0.5, 0.0);
        rho[3] = C64(0.5, 0.0);
        EntropyWorkspace ws;
        std::vector<C64> copy = rho;
        assert(std::abs(entropy_hermitian_inplace(copy, 2, ws) - 1.0) < tol);
        const std::vector<double> sigma{std::sqrt(0.5), std::sqrt(0.5)};
        assert(std::abs(entropy_from_singular_values(sigma) - 1.0) < tol);
    }

    std::cout << "tmi renyi-2 primitives ok\n";
}

} // namespace

int main()
{
    test_circuit_metadata();
    test_linspace();
    test_environment_parsing();
    test_density_round_trip();
    test_sequential_measurement_probabilities();
    test_resume_seed_stream_is_invertible();
    test_resume_csv_row_splitting();
    test_resume_half_entropy_schedule();
    test_resume_running_stats_round_trip();
    test_resume_checkpoint_scanning();
    test_entropy_resume();
    test_tmi_renyi2_primitives();
    std::cout << "core tests passed\n";
}

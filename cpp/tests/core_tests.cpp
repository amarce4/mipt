#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/free_energy_measure.hpp"
#include "mipt/free_energy_resume.hpp"
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
    std::cout << "core tests passed\n";
}

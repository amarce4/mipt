#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/free_energy_measure.hpp"
#include "mipt/types.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
} // namespace

int main()
{
    test_circuit_metadata();
    test_linspace();
    test_environment_parsing();
    test_density_round_trip();
    test_sequential_measurement_probabilities();
    std::cout << "core tests passed\n";
}

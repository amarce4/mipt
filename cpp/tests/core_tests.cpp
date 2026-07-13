#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/types.hpp"

#include <cassert>
#include <cmath>
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
} // namespace

int main()
{
    test_circuit_metadata();
    test_linspace();
    test_environment_parsing();
    test_density_round_trip();
    std::cout << "core tests passed\n";
}

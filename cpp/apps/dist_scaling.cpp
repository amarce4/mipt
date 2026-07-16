// Online two-site mutual-information and negativity distance scaling.

#include "mipt/dist_scaling.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    try
    {
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 ||
             std::strcmp(argv[1], "-h") == 0))
        {
            mipt::dist::print_usage(argv[0]);
            return 0;
        }
        if (argc > 7)
        {
            std::cerr << "Too many arguments. Use --help for usage.\n";
            return 2;
        }

        const int n = (argc > 1) ? std::stoi(argv[1]) : 10;
        const int periods = (argc > 2) ? std::stoi(argv[2]) : 10;
        const double p = (argc > 3) ? std::stod(argv[3]) : 0.17;
        const int realizations = (argc > 4) ? std::stoi(argv[4]) : 10;
        const int circuit_type_value = (argc > 5) ? std::stoi(argv[5]) : 0;
        const mipt::CircuitType circuit_type =
            mipt::parse_circuit_type(circuit_type_value);
        const std::string output_path =
            (argc > 6)
                ? std::string(argv[6])
                : mipt::dist::default_output_path(
                      n, periods, p, realizations, circuit_type);

        mipt::dist::run(n, periods, p, realizations,
                        circuit_type, output_path);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\nerror: " << error.what() << '\n';
        return 1;
    }
}

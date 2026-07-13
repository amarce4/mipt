// Online tripartite-mutual-information simulator.

#include "mipt/tmi/runner.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char **argv)
{
    try
    {
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 ||
             std::strcmp(argv[1], "-h") == 0))
        {
            mipt::tmi::print_usage(argv[0]);
            return 0;
        }
        if (argc > 10)
        {
            std::cerr << "Too many arguments. Use --help for usage.\n";
            return 2;
        }

        const int n = (argc > 1) ? std::stoi(argv[1]) : 10;
        const int periods = (argc > 2) ? std::stoi(argv[2]) : 10;
        const int realizations = (argc > 3) ? std::stoi(argv[3]) : 10;
        const int res = (argc > 4) ? std::stoi(argv[4]) : 5;
        const double p_min = (argc > 5) ? std::stod(argv[5]) : 0.0;
        const double p_max = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const int circ_type_value = (argc > 7) ? std::stoi(argv[7]) : 0;

        bool all_cycles = false;
        std::string output_path;
        bool output_path_explicit = false;
        if (argc > 8)
        {
            const std::string_view arg8(argv[8]);
            if (mipt::tmi::is_bool01_arg(arg8))
            {
                all_cycles = mipt::tmi::parse_bool01_arg(arg8, "all_cycles");
                if (argc > 9)
                {
                    output_path = argv[9];
                    output_path_explicit = true;
                }
            }
            else
            {
                output_path = argv[8];
                output_path_explicit = true;
                if (argc > 9)
                {
                    all_cycles = mipt::tmi::parse_bool01_arg(argv[9], "all_cycles");
                }
            }
        }

        const mipt::CircuitType circuit_mode = mipt::parse_circuit_type(circ_type_value);
        if (!output_path_explicit)
        {
            output_path = mipt::tmi::default_tmi_output_path(n,
                                                          realizations,
                                                          res,
                                                          p_min,
                                                          p_max,
                                                          circuit_mode,
                                                          all_cycles);
        }

        mipt::tmi::run_1d_sim_tmi(n,
                                periods,
                                realizations,
                                res,
                                p_min,
                                p_max,
                                circuit_mode,
                                all_cycles,
                                output_path);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}

// Online distance scaling of two- and three-party entanglement measures.

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
        if (argc > 9)
        {
            std::cerr << "Too many arguments. Use --help for usage.\n";
            return 2;
        }

        mipt::dist::RunConfig config;
        config.k = (argc > 1) ? std::stoi(argv[1]) : 2;
        config.n = (argc > 2) ? std::stoi(argv[2]) : 10;
        config.periods = (argc > 3) ? std::stoi(argv[3]) : 10;
        config.p = (argc > 4) ? std::stod(argv[4]) : 0.17;
        config.realizations = (argc > 5) ? std::stoi(argv[5]) : 10;
        config.type = mipt::parse_circuit_type((argc > 6) ? std::stoi(argv[6]) : 0);
        config.triangle_balance_cutoff =
            (argc > 8) ? std::stod(argv[8])
                       : mipt::env::real("MIPT_DIST_TRIANGLE_BALANCE", 0.5, 0.0, 1.0);
        // An explicit output path is taken as given. Left empty, it is filled in
        // by `run` once the arguments have been validated -- the default depends
        // on all of them, and k=0 needs two paths rather than one.
        if (argc > 7 && argv[7][0] != '\0')
        {
            config.output_path = std::string(argv[7]);
        }

        mipt::dist::run(config);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\nerror: " << error.what() << '\n';
        return 1;
    }
}

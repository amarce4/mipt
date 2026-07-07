// entropy.cpp
// Online entanglement-entropy scan for 1D MIPT trajectories.
//
// Usage:
//   ./entropy.exe [N] [periods] [p] [realizations] [fermion] [output_csv]

#define main mipt_existing_main_for_entropy_build
#include "mipt_fermion.cpp"
#undef main

#include <dlfcn.h>
#include <iomanip>
#include <memory>
#include <sstream>

namespace
{
    using Complex = std::complex<double>;

    using ZheevdFn = void (*)(char *, char *, int *, Complex *, int *, double *,
                              Complex *, int *, double *, int *, int *, int *, int *);

    class LapackRuntime
    {
      public:
        LapackRuntime()
        {
            const char *libs[] = {
                "libopenblas.so.0",
                "libopenblas.so",
                "liblapack.so.3",
                "liblapack.so",
                "libmkl_rt.so",
                nullptr,
            };

            for (const char **lib = libs; *lib != nullptr; ++lib)
            {
                handle_ = dlopen(*lib, RTLD_LAZY | RTLD_LOCAL);
                if (handle_ != nullptr)
                {
                    library_name_ = *lib;
                    break;
                }
            }

            if (handle_ == nullptr)
            {
                throw std::runtime_error(
                    "entropy.exe requires LAPACK at runtime. Install OpenBLAS/LAPACK "
                    "or ensure libopenblas.so/liblapack.so is on LD_LIBRARY_PATH.");
            }

            zheevd_ = reinterpret_cast<ZheevdFn>(dlsym(handle_, "zheevd_"));
            if (zheevd_ == nullptr)
            {
                zheevd_ = reinterpret_cast<ZheevdFn>(dlsym(handle_, "zheevd"));
            }
            if (zheevd_ == nullptr)
            {
                throw std::runtime_error(
                    "Loaded " + library_name_ + " but could not find LAPACK symbol zheevd_.");
            }
        }

        LapackRuntime(const LapackRuntime &) = delete;
        LapackRuntime &operator=(const LapackRuntime &) = delete;

        ~LapackRuntime()
        {
            if (handle_ != nullptr)
            {
                dlclose(handle_);
            }
        }

        const std::string &library_name() const { return library_name_; }

        std::vector<double> hermitian_eigenvalues(std::vector<Complex> matrix_col_major,
                                                  int dim) const
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("Cannot diagonalize an empty matrix.");
            }
            if (matrix_col_major.size() != static_cast<std::size_t>(dim) *
                                               static_cast<std::size_t>(dim))
            {
                throw std::invalid_argument("Hermitian matrix payload has the wrong size.");
            }

            // Remove harmless roundoff asymmetry before calling the Hermitian solver.
            for (int i = 0; i < dim; ++i)
            {
                matrix_col_major[static_cast<std::size_t>(i) * dim + i] =
                    Complex(std::real(matrix_col_major[static_cast<std::size_t>(i) * dim + i]), 0.0);
                for (int j = i + 1; j < dim; ++j)
                {
                    const std::size_t a = static_cast<std::size_t>(i) +
                                          static_cast<std::size_t>(j) * dim;
                    const std::size_t b = static_cast<std::size_t>(j) +
                                          static_cast<std::size_t>(i) * dim;
                    const Complex avg = 0.5 * (matrix_col_major[a] + std::conj(matrix_col_major[b]));
                    matrix_col_major[a] = avg;
                    matrix_col_major[b] = std::conj(avg);
                }
            }

            std::vector<double> eigenvalues(static_cast<std::size_t>(dim), 0.0);

            char jobz = 'N';
            char uplo = 'L';
            int n = dim;
            int lda = dim;
            int info = 0;
            int lwork = -1;
            int lrwork = -1;
            int liwork = -1;
            Complex work_query{};
            double rwork_query = 0.0;
            int iwork_query = 0;

            zheevd_(&jobz, &uplo, &n, matrix_col_major.data(), &lda,
                    eigenvalues.data(), &work_query, &lwork,
                    &rwork_query, &lrwork, &iwork_query, &liwork, &info);
            if (info != 0)
            {
                throw std::runtime_error("LAPACK zheevd workspace query failed with info=" +
                                         std::to_string(info));
            }

            lwork = std::max(1, static_cast<int>(std::ceil(std::real(work_query))));
            lrwork = std::max(1, static_cast<int>(std::ceil(rwork_query)));
            liwork = std::max(1, iwork_query);

            std::vector<Complex> work(static_cast<std::size_t>(lwork));
            std::vector<double> rwork(static_cast<std::size_t>(lrwork));
            std::vector<int> iwork(static_cast<std::size_t>(liwork));

            zheevd_(&jobz, &uplo, &n, matrix_col_major.data(), &lda,
                    eigenvalues.data(), work.data(), &lwork,
                    rwork.data(), &lrwork, iwork.data(), &liwork, &info);
            if (info != 0)
            {
                throw std::runtime_error("LAPACK zheevd diagonalization failed with info=" +
                                         std::to_string(info));
            }

            return eigenvalues;
        }

      private:
        void *handle_ = nullptr;
        ZheevdFn zheevd_ = nullptr;
        std::string library_name_;
    };

    std::vector<Complex> cudaq_state_to_host_complex128(cudaq::state &state, int n)
    {
        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
        {
            throw std::runtime_error("Expected a contiguous rank-1 CUDA-Q statevector tensor.");
        }

        std::vector<Complex> host_state(dim);

        if (state.is_on_gpu())
        {
            if (precision == cudaq::SimulationState::precision::fp64)
            {
                std::vector<std::complex<double>> tmp(dim);
                state.to_host(tmp.data(), tmp.size());
                std::copy(tmp.begin(), tmp.end(), host_state.begin());
            }
            else
            {
                std::vector<std::complex<float>> tmp(dim);
                state.to_host(tmp.data(), tmp.size());
                for (std::size_t i = 0; i < dim; ++i)
                {
                    host_state[i] = Complex(static_cast<double>(std::real(tmp[i])),
                                            static_cast<double>(std::imag(tmp[i])));
                }
            }
            return host_state;
        }

        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            const auto *ptr = reinterpret_cast<const std::complex<double> *>(tensor.data);
            std::copy(ptr, ptr + dim, host_state.begin());
        }
        else
        {
            const auto *ptr = reinterpret_cast<const std::complex<float> *>(tensor.data);
            for (std::size_t i = 0; i < dim; ++i)
            {
                host_state[i] = Complex(static_cast<double>(std::real(ptr[i])),
                                        static_cast<double>(std::imag(ptr[i])));
            }
        }

        return host_state;
    }

    std::vector<int> cyclic_subsystem(int n, int start, int kept)
    {
        if (kept <= 0 || kept > n)
        {
            throw std::invalid_argument("Invalid cyclic subsystem size.");
        }
        std::vector<int> modes;
        modes.reserve(static_cast<std::size_t>(kept));
        for (int i = 0; i < kept; ++i)
        {
            modes.push_back((start + i) % n);
        }
        return modes;
    }

    std::vector<Complex> reduced_density_col_major_qubit(
        const std::vector<Complex> &psi,
        int n,
        const std::vector<int> &kept_modes)
    {
        const int kept = static_cast<int>(kept_modes.size());
        const std::uint64_t retained_dim_u64 = checked_pow2_u64(kept);
        const std::uint64_t env_dim_u64 = checked_pow2_u64(n - kept);
        const std::size_t retained_dim = static_cast<std::size_t>(retained_dim_u64);
        const std::size_t env_dim = static_cast<std::size_t>(env_dim_u64);

        std::vector<unsigned char> is_retained(static_cast<std::size_t>(n), 0);
        for (int q : kept_modes)
        {
            if (q < 0 || q >= n)
            {
                throw std::invalid_argument("Subsystem index is out of range.");
            }
            if (is_retained[static_cast<std::size_t>(q)])
            {
                throw std::invalid_argument("Subsystem contains duplicate sites.");
            }
            is_retained[static_cast<std::size_t>(q)] = 1;
        }

        std::vector<int> environment_modes;
        environment_modes.reserve(static_cast<std::size_t>(n - kept));
        for (int q = 0; q < n; ++q)
        {
            if (!is_retained[static_cast<std::size_t>(q)])
            {
                environment_modes.push_back(q);
            }
        }

        std::vector<std::uint64_t> retained_offsets(retained_dim, 0);
        for (std::size_t local = 0; local < retained_dim; ++local)
        {
            std::uint64_t offset = 0;
            for (int bit = 0; bit < kept; ++bit)
            {
                if ((local >> bit) & std::size_t{1})
                {
                    offset |= std::uint64_t{1} << kept_modes[static_cast<std::size_t>(bit)];
                }
            }
            retained_offsets[local] = offset;
        }

        std::vector<Complex> rho(retained_dim * retained_dim, Complex(0.0, 0.0));

        for (std::uint64_t env = 0; env < env_dim_u64; ++env)
        {
            std::uint64_t base = 0;
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                if ((env >> e) & std::uint64_t{1})
                {
                    base |= std::uint64_t{1} << environment_modes[e];
                }
            }

            for (std::size_t row = 0; row < retained_dim; ++row)
            {
                const Complex a = psi[static_cast<std::size_t>(base | retained_offsets[row])];
                for (std::size_t col = 0; col < retained_dim; ++col)
                {
                    const Complex b = psi[static_cast<std::size_t>(base | retained_offsets[col])];
                    rho[col * retained_dim + row] += a * std::conj(b); // column-major
                }
            }
        }

        return rho;
    }

    int fermionic_reorder_sign_from_index(std::uint64_t occupation_mask,
                                           const std::vector<int> &position_in_new_order,
                                           int n)
    {
        // Equivalent to fermionic_reorder_sign_from_occupations(...) in
        // mipt_fermion.cpp, but avoids constructing an occupation vector for
        // every basis state. Iterate occupied modes in canonical order and
        // count inversions of their positions in the requested tensor order.
        std::uint64_t seen_new_positions = 0;
        int parity = 0;
        for (int mode = 0; mode < n; ++mode)
        {
            if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
            {
                continue;
            }
            const int pos = position_in_new_order[static_cast<std::size_t>(mode)];
            const std::uint64_t lower_or_equal =
                (pos >= 63) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (pos + 1)) - 1u);
            const std::uint64_t higher_seen = seen_new_positions & ~lower_or_equal;
            parity ^= (__builtin_popcountll(higher_seen) & 1);
            seen_new_positions |= (std::uint64_t{1} << pos);
        }
        return parity ? -1 : 1;
    }

    std::vector<Complex> reduced_density_col_major_fermion(
        const std::vector<Complex> &psi,
        int n,
        const std::vector<int> &kept_modes)
    {
        const int kept = static_cast<int>(kept_modes.size());
        const std::uint64_t retained_dim_u64 = checked_pow2_u64(kept);
        const std::uint64_t env_dim_u64 = checked_pow2_u64(n - kept);
        const std::size_t retained_dim = static_cast<std::size_t>(retained_dim_u64);

        std::vector<unsigned char> is_retained(static_cast<std::size_t>(n), 0);
        for (int q : kept_modes)
        {
            if (q < 0 || q >= n)
            {
                throw std::invalid_argument("Subsystem index is out of range.");
            }
            if (is_retained[static_cast<std::size_t>(q)])
            {
                throw std::invalid_argument("Subsystem contains duplicate sites.");
            }
            is_retained[static_cast<std::size_t>(q)] = 1;
        }

        std::vector<int> environment_modes;
        environment_modes.reserve(static_cast<std::size_t>(n - kept));
        for (int q = 0; q < n; ++q)
        {
            if (!is_retained[static_cast<std::size_t>(q)])
            {
                environment_modes.push_back(q);
            }
        }

        std::vector<int> new_mode_order = kept_modes;
        new_mode_order.insert(new_mode_order.end(), environment_modes.begin(), environment_modes.end());

        std::vector<int> position_in_new_order(static_cast<std::size_t>(n), 0);
        for (std::size_t pos = 0; pos < new_mode_order.size(); ++pos)
        {
            position_in_new_order[static_cast<std::size_t>(new_mode_order[pos])] = static_cast<int>(pos);
        }

        std::vector<std::uint64_t> retained_offsets(retained_dim, 0);
        for (std::size_t local = 0; local < retained_dim; ++local)
        {
            std::uint64_t offset = 0;
            for (int bit = 0; bit < kept; ++bit)
            {
                if ((local >> bit) & std::size_t{1})
                {
                    offset |= std::uint64_t{1} << kept_modes[static_cast<std::size_t>(bit)];
                }
            }
            retained_offsets[local] = offset;
        }

        std::vector<Complex> rho(retained_dim * retained_dim, Complex(0.0, 0.0));
        std::vector<Complex> amplitudes(retained_dim);

        for (std::uint64_t env = 0; env < env_dim_u64; ++env)
        {
            std::uint64_t base = 0;
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                if ((env >> e) & std::uint64_t{1})
                {
                    base |= std::uint64_t{1} << environment_modes[e];
                }
            }

            for (std::size_t local = 0; local < retained_dim; ++local)
            {
                const std::uint64_t old_index = base | retained_offsets[local];
                const int sign = fermionic_reorder_sign_from_index(old_index, position_in_new_order, n);
                amplitudes[local] = psi[static_cast<std::size_t>(old_index)] * static_cast<double>(sign);
            }

            for (std::size_t row = 0; row < retained_dim; ++row)
            {
                const Complex a = amplitudes[row];
                for (std::size_t col = 0; col < retained_dim; ++col)
                {
                    rho[col * retained_dim + row] += a * std::conj(amplitudes[col]); // column-major
                }
            }
        }

        return rho;
    }

    double von_neumann_entropy_bits(const LapackRuntime &lapack,
                                    std::vector<Complex> rho_col_major,
                                    int dim)
    {
        std::vector<double> eigenvalues = lapack.hermitian_eigenvalues(std::move(rho_col_major), dim);

        double positive_trace = 0.0;
        for (double lambda : eigenvalues)
        {
            if (lambda > 0.0)
            {
                positive_trace += lambda;
            }
        }
        if (!(positive_trace > 0.0))
        {
            return 0.0;
        }

        double entropy = 0.0;
        constexpr double tiny_negative_tol = -1e-10;
        for (double lambda : eigenvalues)
        {
            if (lambda < tiny_negative_tol)
            {
                std::cerr << "Warning: negative reduced-density eigenvalue " << lambda
                          << " encountered; clamping to zero.\n";
            }
            if (lambda <= 0.0)
            {
                continue;
            }
            const double normalized_lambda = lambda / positive_trace;
            entropy -= normalized_lambda * (std::log(normalized_lambda) / std::log(2.0));
        }
        if (entropy < 0.0 && entropy > -1e-10)
        {
            entropy = 0.0;
        }
        return entropy;
    }

    cudaq::state simulate_trajectory(int n, int periods, double p, int fermion)
    {
        if (fermion == 0)
        {
            auto layers = mipt_frontend(n, periods, p);
            return cudaq::get_state(MIPTKernel_1D{}, n, layers, true);
        }
        if (fermion == 1)
        {
            auto layers = mipt_fermion_frontend(n, periods, p, true);
            return cudaq::get_state(MIPTFermionKernel_1D{}, n, layers);
        }
        if (fermion == 2)
        {
            auto layers = frgs_mipt_frontend(n, periods, p);
            return cudaq::get_state(MIPTKernel_1D_FRGS{}, n, layers, true);
        }
        throw std::invalid_argument("fermion must be 0, 1, or 2.");
    }

    void write_header(std::ofstream &csv, int max_kept)
    {
        csv << "p";
        for (int kept = 2; kept <= max_kept; ++kept)
        {
            csv << ",entropy_" << kept;
        }
        csv << "\n";
        csv.flush();
    }

    void print_usage(const char *argv0)
    {
        std::cout << "Usage: " << argv0
                  << " [N] [periods] [p] [realizations] [fermion] [output_csv]\n\n"
                  << "  N            number of sites/qubits/modes in the 1D ring\n"
                  << "  periods      number of circuit periods/layers to simulate\n"
                  << "  p            measurement rate in [0,1]\n"
                  << "  realizations number of circuit trajectories to simulate\n"
                  << "  fermion      0 = qubit/MMS, 1 = parity-preserving Haar fermion, 2 = FRGS fermion\n"
                  << "  output_csv   output path; columns are p,entropy_2,...,entropy_{floor(N/2)}\n\n"
                  << "Notes:\n"
                  << "  Entropies use log2 and are therefore in bits.\n"
                  << "  Rows are ordered by realization, then cyclic start index 0..N-1.\n";
    }
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))
        {
            print_usage(argv[0]);
            return 0;
        }

        if (argc != 7)
        {
            print_usage(argv[0]);
            return 2;
        }

        const int n = std::stoi(argv[1]);
        const int periods = std::stoi(argv[2]);
        const double p = std::stod(argv[3]);
        const int realizations = std::stoi(argv[4]);
        const int fermion = std::stoi(argv[5]);
        const std::string output_csv = argv[6];

        if (n < 4)
        {
            throw std::invalid_argument("N must be at least 4 to produce size-2 through N/2 RDMs.");
        }
        if (periods <= 0)
        {
            throw std::invalid_argument("periods must be positive.");
        }
        if (p < 0.0 || p > 1.0)
        {
            throw std::invalid_argument("p must be in [0,1].");
        }
        if (realizations <= 0)
        {
            throw std::invalid_argument("realizations must be positive.");
        }
        if (fermion < 0 || fermion > 2)
        {
            throw std::invalid_argument("fermion must be 0, 1, or 2.");
        }
        if (fermion != 0 && (n % 2) != 0)
        {
            throw std::invalid_argument("Fermionic circuit modes require even N.");
        }

        const int max_kept = n / 2;

        LapackRuntime lapack;
        std::ofstream csv(output_csv, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            throw std::runtime_error("Could not open output CSV: " + output_csv);
        }
        csv << std::setprecision(17);
        write_header(csv, max_kept);

        std::cout << "Running entropy scan: N=" << n
                  << ", p=" << p
                  << ", realizations=" << realizations
                  << ", fermion=" << fermion
                  << ", periods=" << periods
                  << ", LAPACK=" << lapack.library_name() << "\n";
        std::cout << "Writing " << static_cast<long long>(realizations) * n
                  << " CSV rows to " << output_csv << ".\n";

        double sec_per_circ = 0.0;
        const auto total_start = std::chrono::steady_clock::now();

        for (int r = 0; r < realizations; ++r)
        {
            const auto circ_start = std::chrono::steady_clock::now();
            auto state = simulate_trajectory(n, periods, p, fermion);
            const std::vector<Complex> psi = cudaq_state_to_host_complex128(state, n);

            for (int start = 0; start < n; ++start)
            {
                csv << p;
                for (int kept = 2; kept <= max_kept; ++kept)
                {
                    const std::vector<int> subsystem = cyclic_subsystem(n, start, kept);
                    std::vector<Complex> rho = (fermion == 0)
                                                   ? reduced_density_col_major_qubit(psi, n, subsystem)
                                                   : reduced_density_col_major_fermion(psi, n, subsystem);
                    const int dim = static_cast<int>(checked_pow2_u64(kept));
                    const double entropy = von_neumann_entropy_bits(lapack, std::move(rho), dim);
                    csv << ',' << entropy;
                }
                csv << '\n';
                csv.flush();
            }

            const auto circ_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> dt = circ_end - circ_start;
            if (r == 0)
            {
                sec_per_circ = dt.count();
            }
            else
            {
                sec_per_circ = (sec_per_circ * r + dt.count()) / (r + 1);
            }

            const int remaining = realizations - r - 1;
            const double cps = (sec_per_circ > 0.0) ? (1.0 / sec_per_circ) : 0.0;
            std::cout << "\rCompleted " << (r + 1) << "/" << realizations
                      << " trajectories; average circuits/second="
                      << std::round(cps * 100.0) / 100.0
                      << "; ETA=" << static_cast<int>(remaining * sec_per_circ)
                      << "s          " << std::flush;
        }

        const auto total_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> total_dt = total_end - total_start;
        std::cout << "\nSaved entropy CSV to " << output_csv
                  << ". Elapsed time: " << total_dt.count() << "s\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "entropy.exe error: " << e.what() << "\n";
        return 1;
    }
}

#pragma once

#include "mipt/circuit.hpp"
#include "mipt/env.hpp"
#include "mipt/rdm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::native_mps
{
    using C = std::complex<double>;
    using Gate2 = std::array<C, 4>;
    using Gate4 = std::array<C, 16>;

    inline bool enabled()
    {
        return env::boolean("MIPT_NATIVE_MPS", false) || env::boolean("MIPT_NATIVE_TEBD", false);
    }

    inline int max_bond()
    {
        long fallback = env::integer("CUDAQ_MPS_MAX_BOND", 64, 1, 4096);
        return static_cast<int>(env::integer("MIPT_NATIVE_MPS_MAX_BOND", fallback, 1, 4096));
    }

    inline double abs_cutoff()
    {
        double fallback = env::real("CUDAQ_MPS_ABS_CUTOFF", 1.0e-10, 0.0, 1.0);
        return env::real("MIPT_NATIVE_MPS_ABS_CUTOFF", fallback, 0.0, 1.0);
    }

    inline double rel_cutoff()
    {
        double fallback = env::real("CUDAQ_MPS_RELATIVE_CUTOFF", 1.0e-10, 0.0, 1.0);
        return env::real("MIPT_NATIVE_MPS_RELATIVE_CUTOFF", fallback, 0.0, 1.0);
    }

    inline bool verbose()
    {
        return env::boolean("MIPT_VERBOSE", false) || env::boolean("MIPT_NATIVE_MPS_VERBOSE", false);
    }

    inline int prefix_layers()
    {
        return static_cast<int>(env::integer("MIPT_DEBUG_PREFIX_LAYERS", 0, 0, 1000000));
    }

    inline int normalize_interval()
    {
        return static_cast<int>(env::integer("MIPT_NATIVE_MPS_NORMALIZE_INTERVAL", 1, 1, 1000000));
    }

    inline void log(const std::string &msg)
    {
        if (verbose()) std::cerr << "[native-mps] " << msg << '\n';
    }

    namespace lapack
    {
        using zgesvd_fn = void (*)(char *, char *, int *, int *, C *, int *, double *, C *, int *, C *, int *, C *, int *, double *, int *);

        struct Handle
        {
            bool initialized = false;
            std::vector<void *> libs;
            zgesvd_fn zgesvd = nullptr;
        };

        inline Handle &handle()
        {
            static Handle h;
            return h;
        }

        inline void init()
        {
            auto &h = handle();
            if (h.initialized) return;
            h.initialized = true;
            const char *names[] = {
                "libopenblas.so.0", "libopenblas.so", "liblapack.so.3", "liblapack.so", "libblas.so.3", "libblas.so"};
            for (const char *name : names)
            {
                void *lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
                if (!lib) continue;
                h.libs.push_back(lib);
                if (!h.zgesvd) h.zgesvd = reinterpret_cast<zgesvd_fn>(dlsym(lib, "zgesvd_"));
                if (h.zgesvd) break;
            }
        }

        inline zgesvd_fn zgesvd()
        {
            init();
            return handle().zgesvd;
        }
    }

    inline Gate4 swap_gate();
    inline Gate4 fswap_gate();

    struct Tensor
    {
        int dl = 1;
        int dr = 1;
        std::vector<C> data; // ((left * 2 + phys) * dr + right)

        Tensor() : dl(1), dr(1), data(2, C(0.0, 0.0)) {}
        Tensor(int dl_, int dr_) : dl(dl_), dr(dr_), data(static_cast<std::size_t>(dl_) * 2u * static_cast<std::size_t>(dr_), C(0.0, 0.0)) {}

        C &operator()(int l, int s, int r)
        {
            return data[(static_cast<std::size_t>(l) * 2u + static_cast<std::size_t>(s)) * static_cast<std::size_t>(dr) + static_cast<std::size_t>(r)];
        }
        const C &operator()(int l, int s, int r) const
        {
            return data[(static_cast<std::size_t>(l) * 2u + static_cast<std::size_t>(s)) * static_cast<std::size_t>(dr) + static_cast<std::size_t>(r)];
        }
    };

    class MPS
    {
      public:
        explicit MPS(int n_, int chi_, double abs_cut_, double rel_cut_)
            : n(n_), chi_max(chi_), abs_cut(abs_cut_), rel_cut(rel_cut_), a(static_cast<std::size_t>(n_))
        {
            for (int i = 0; i < n; ++i)
            {
                a[static_cast<std::size_t>(i)] = Tensor(1, 1);
                a[static_cast<std::size_t>(i)](0, 0, 0) = C(1.0, 0.0);
                a[static_cast<std::size_t>(i)](0, 1, 0) = C(0.0, 0.0);
            }
        }

        void apply_one_site(int site, const Gate2 &u)
        {
            Tensor &t = a.at(static_cast<std::size_t>(site));
            std::vector<C> out(t.data.size(), C(0.0, 0.0));
            for (int l = 0; l < t.dl; ++l)
            {
                for (int r = 0; r < t.dr; ++r)
                {
                    const C v0 = t(l, 0, r);
                    const C v1 = t(l, 1, r);
                    out[(static_cast<std::size_t>(l) * 2u + 0u) * static_cast<std::size_t>(t.dr) + static_cast<std::size_t>(r)] = u[0] * v0 + u[1] * v1;
                    out[(static_cast<std::size_t>(l) * 2u + 1u) * static_cast<std::size_t>(t.dr) + static_cast<std::size_t>(r)] = u[2] * v0 + u[3] * v1;
                }
            }
            t.data.swap(out);
        }

        void apply_two_site(int q0, int q1, const Gate4 &u)
        {
            if (q0 == q1 || q0 < 0 || q1 < 0 || q0 >= n || q1 >= n)
            {
                throw std::invalid_argument("Invalid native-MPS two-site gate target.");
            }
            if (std::abs(q0 - q1) == 1)
            {
                const int left = std::min(q0, q1);
                Gate4 gate = (q0 < q1) ? u : swap_gate_order(u);
                apply_adjacent(left, gate);
                return;
            }
            apply_long_range(q0, q1, u);
        }

        double norm() const
        {
            std::vector<C> env(1, C(1.0, 0.0));
            int dim = 1;
            for (int site = 0; site < n; ++site)
            {
                const Tensor &t = a[static_cast<std::size_t>(site)];
                std::vector<C> next(
                    static_cast<std::size_t>(t.dr) * static_cast<std::size_t>(t.dr),
                    C(0.0, 0.0));
                for (int lb = 0; lb < t.dl; ++lb)
                {
                    for (int lk = 0; lk < t.dl; ++lk)
                    {
                        const C e = env[static_cast<std::size_t>(lb) *
                                            static_cast<std::size_t>(dim) +
                                        static_cast<std::size_t>(lk)];
                        if (std::abs(e) == 0.0) continue;
                        for (int s = 0; s < 2; ++s)
                        {
                            for (int rb = 0; rb < t.dr; ++rb)
                            {
                                const C vb = t(lb, s, rb);
                                if (std::abs(vb) == 0.0) continue;
                                for (int rk = 0; rk < t.dr; ++rk)
                                {
                                    next[static_cast<std::size_t>(rb) *
                                             static_cast<std::size_t>(t.dr) +
                                         static_cast<std::size_t>(rk)] +=
                                        e * vb * std::conj(t(lk, s, rk));
                                }
                            }
                        }
                    }
                }
                env.swap(next);
                dim = t.dr;
            }
            return std::max(0.0, env.empty() ? 0.0 : env[0].real());
        }

        void normalize()
        {
            const double nrm = norm();
            if (!(nrm > 0.0) || !std::isfinite(nrm))
            {
                throw std::runtime_error("Native MPS norm became non-positive/non-finite.");
            }
            const double scale = 1.0 / std::sqrt(nrm);
            for (auto &x : a[0].data) x *= scale;
        }

        double prob_one(int site) const
        {
            std::array<int, 1> keep{{site}};
            std::array<int, 1> row{{1}};
            std::array<int, 1> col{{1}};
            const C v = contract_fixed(keep.data(), row.data(), col.data(), 1, false, 1.0);
            const double nrm = norm();
            if (nrm <= 0.0) return 0.0;
            return std::clamp(v.real() / nrm, 0.0, 1.0);
        }

        void measure_z(int site, std::mt19937 &rng)
        {
            const double p1 = prob_one(site);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            const int outcome = (uniform(rng) < p1) ? 1 : 0;
            Tensor &t = a.at(static_cast<std::size_t>(site));
            for (int l = 0; l < t.dl; ++l)
            {
                for (int r = 0; r < t.dr; ++r)
                {
                    t(l, 1 - outcome, r) = C(0.0, 0.0);
                }
            }
            normalize();
        }

        void measure_layer(const std::vector<int> &flags, std::mt19937 &rng)
        {
            for (int q = 0; q < n; ++q)
            {
                if (flags[static_cast<std::size_t>(q)]) measure_z(q, rng);
            }
        }

        void rho3(const Subsystem &subsystem, double *out_ri, bool fermion_trace) const
        {
            std::fill(out_ri, out_ri + 2u * 8u * 8u, 0.0);
            const double nrm = norm();
            if (!(nrm > 0.0)) throw std::runtime_error("Native MPS cannot compute RDM from zero norm.");
            int keep[3] = {subsystem[0], subsystem[1], subsystem[2]};
            for (int row = 0; row < 8; ++row)
            {
                int row_bits[3] = {(row >> 0) & 1, (row >> 1) & 1, (row >> 2) & 1};
                for (int col = 0; col < 8; ++col)
                {
                    int col_bits[3] = {(col >> 0) & 1, (col >> 1) & 1, (col >> 2) & 1};
                    double fixed_sign = 1.0;
                    std::vector<unsigned char> parity_twist(static_cast<std::size_t>(n), 0);
                    if (fermion_trace)
                    {
                        fixed_sign = static_cast<double>(
                            fermion_internal_sign(keep, row_bits) *
                            fermion_internal_sign(keep, col_bits));
                        for (int k = 0; k < 3; ++k)
                        {
                            if ((row_bits[k] ^ col_bits[k]) == 0) continue;
                            for (int q = 0; q < n; ++q)
                            {
                                if (q == keep[0] || q == keep[1] || q == keep[2])
                                {
                                    continue;
                                }
                                if (q < keep[k]) parity_twist[static_cast<std::size_t>(q)] ^= 1u;
                            }
                        }
                    }
                    const C value =
                        contract_fixed(keep, row_bits, col_bits, 3,
                                       fermion_trace, fixed_sign,
                                       parity_twist.empty() ? nullptr
                                                            : parity_twist.data()) /
                        nrm;
                    const std::size_t off = 2u * static_cast<std::size_t>(row * 8 + col);
                    out_ri[off + 0] = value.real();
                    out_ri[off + 1] = value.imag();
                }
            }
        }

        int max_observed_bond() const
        {
            int m = 1;
            for (const auto &t : a) m = std::max(m, std::max(t.dl, t.dr));
            return m;
        }

      private:
        int n = 0;
        int chi_max = 64;
        double abs_cut = 1e-10;
        double rel_cut = 1e-10;
        std::vector<Tensor> a;

        static Gate4 swap_gate_order(const Gate4 &u)
        {
            Gate4 out{};
            const int perm[4] = {0, 2, 1, 3};
            for (int r = 0; r < 4; ++r)
            {
                for (int c = 0; c < 4; ++c)
                {
                    out[static_cast<std::size_t>(r * 4 + c)] = u[static_cast<std::size_t>(perm[r] * 4 + perm[c])];
                }
            }
            return out;
        }

        void apply_long_range(int q0, int q1, const Gate4 &u)
        {
            if (q0 < q1)
            {
                for (int k = q1 - 1; k > q0; --k) apply_adjacent(k, swap_gate());
                apply_adjacent(q0, u);
                for (int k = q0 + 1; k < q1; ++k) apply_adjacent(k, swap_gate());
            }
            else
            {
                // Bring q0 to q1+1.  The adjacent site order is then (q1,q0),
                // so use the gate transformed into the swapped local ordering.
                for (int k = q0 - 1; k > q1; --k) apply_adjacent(k, swap_gate());
                apply_adjacent(q1, swap_gate_order(u));
                for (int k = q1 + 1; k < q0; ++k) apply_adjacent(k, swap_gate());
            }
        }

        void apply_adjacent(int left, const Gate4 &u)
        {
            if (left < 0 || left + 1 >= n) throw std::invalid_argument("Invalid adjacent native-MPS gate target.");
            Tensor &A = a[static_cast<std::size_t>(left)];
            Tensor &B = a[static_cast<std::size_t>(left + 1)];
            if (A.dr != B.dl) throw std::runtime_error("Native MPS bond dimension mismatch.");
            const int dl = A.dl;
            const int bond = A.dr;
            const int dr = B.dr;
            const int rows = dl * 2;
            const int cols = 2 * dr;

            std::vector<C> matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), C(0.0, 0.0));
            for (int l = 0; l < dl; ++l)
            {
                for (int r = 0; r < dr; ++r)
                {
                    C theta[4] = {C(0.0, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(0.0, 0.0)};
                    for (int b = 0; b < bond; ++b)
                    {
                        theta[0] += A(l, 0, b) * B(b, 0, r);
                        theta[1] += A(l, 1, b) * B(b, 0, r);
                        theta[2] += A(l, 0, b) * B(b, 1, r);
                        theta[3] += A(l, 1, b) * B(b, 1, r);
                    }
                    C out[4]{};
                    for (int rr = 0; rr < 4; ++rr)
                    {
                        for (int cc = 0; cc < 4; ++cc)
                        {
                            out[rr] += u[static_cast<std::size_t>(rr * 4 + cc)] * theta[cc];
                        }
                    }
                    // row basis = left physical bit + 2*right physical bit.
                    matrix[static_cast<std::size_t>(l * 2 + 0) + static_cast<std::size_t>(0 * dr + r) * static_cast<std::size_t>(rows)] = out[0];
                    matrix[static_cast<std::size_t>(l * 2 + 1) + static_cast<std::size_t>(0 * dr + r) * static_cast<std::size_t>(rows)] = out[1];
                    matrix[static_cast<std::size_t>(l * 2 + 0) + static_cast<std::size_t>(1 * dr + r) * static_cast<std::size_t>(rows)] = out[2];
                    matrix[static_cast<std::size_t>(l * 2 + 1) + static_cast<std::size_t>(1 * dr + r) * static_cast<std::size_t>(rows)] = out[3];
                }
            }

            svd_split(matrix, rows, cols, dl, dr, A, B);
        }

        void svd_split(std::vector<C> &m_col_major, int rows, int cols, int dl, int dr, Tensor &left_tensor, Tensor &right_tensor)
        {
            auto zgesvd = lapack::zgesvd();
            if (!zgesvd)
            {
                throw std::runtime_error("Native MPS requires LAPACK zgesvd_ at runtime. Install OpenBLAS/LAPACK or disable MIPT_NATIVE_MPS.");
            }
            int m = rows;
            int ncols = cols;
            int lda = rows;
            const int k = std::min(rows, cols);
            int ldu = rows;
            int ldvt = k;
            std::vector<double> s(static_cast<std::size_t>(k), 0.0);
            std::vector<C> u(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), C(0.0, 0.0));
            std::vector<C> vt(static_cast<std::size_t>(k) * static_cast<std::size_t>(cols), C(0.0, 0.0));
            std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 5 * k)), 0.0);
            int lwork = -1;
            int info = 0;
            C work_query(0.0, 0.0);
            char job = 'S';
            zgesvd(&job, &job, &m, &ncols, m_col_major.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, &work_query, &lwork, rwork.data(), &info);
            if (info != 0) throw std::runtime_error("Native MPS LAPACK zgesvd workspace query failed.");
            lwork = std::max(1, static_cast<int>(std::ceil(work_query.real())));
            std::vector<C> work(static_cast<std::size_t>(lwork), C(0.0, 0.0));
            zgesvd(&job, &job, &m, &ncols, m_col_major.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
            if (info != 0) throw std::runtime_error("Native MPS LAPACK zgesvd failed.");

            int keep = 0;
            const double s0 = (k > 0) ? s[0] : 0.0;
            for (int i = 0; i < k && keep < chi_max; ++i)
            {
                const bool keep_abs = s[i] > abs_cut;
                const bool keep_rel = (s0 <= 0.0) ? (i == 0) : (s[i] / s0 > rel_cut);
                if (i == 0 || (keep_abs && keep_rel)) ++keep;
            }
            keep = std::max(1, keep);

            Tensor L(dl, keep);
            Tensor R(keep, dr);
            for (int l = 0; l < dl; ++l)
            {
                for (int sp = 0; sp < 2; ++sp)
                {
                    const int row = l * 2 + sp;
                    for (int aidx = 0; aidx < keep; ++aidx)
                    {
                        L(l, sp, aidx) = u[static_cast<std::size_t>(row) + static_cast<std::size_t>(aidx) * static_cast<std::size_t>(rows)];
                    }
                }
            }
            for (int aidx = 0; aidx < keep; ++aidx)
            {
                for (int sp = 0; sp < 2; ++sp)
                {
                    for (int r = 0; r < dr; ++r)
                    {
                        const int col = sp * dr + r;
                        R(aidx, sp, r) =
                            s[static_cast<std::size_t>(aidx)] *
                            vt[static_cast<std::size_t>(aidx) +
                               static_cast<std::size_t>(col) *
                                   static_cast<std::size_t>(k)];
                    }
                }
            }
            left_tensor = std::move(L);
            right_tensor = std::move(R);
        }

        static int fermion_internal_sign(const int keep[3], const int bits[3])
        {
            int parity = 0;
            for (int a = 0; a < 3; ++a)
            {
                if (!bits[a]) continue;
                for (int b = a + 1; b < 3; ++b)
                {
                    if (bits[b] && keep[a] > keep[b]) parity ^= 1;
                }
            }
            return parity ? -1 : 1;
        }

        C contract_fixed(const int *keep, const int *row_bits,
                         const int *col_bits, int kept_count, bool use_twist,
                         double fixed_sign,
                         const unsigned char *twist = nullptr) const
        {
            std::vector<int> keep_pos(static_cast<std::size_t>(n), -1);
            for (int k = 0; k < kept_count; ++k)
            {
                keep_pos[static_cast<std::size_t>(keep[k])] = k;
            }
            std::vector<C> env(1, C(fixed_sign, 0.0));
            int dim = 1;
            for (int site = 0; site < n; ++site)
            {
                const Tensor &t = a[static_cast<std::size_t>(site)];
                std::vector<C> next(
                    static_cast<std::size_t>(t.dr) * static_cast<std::size_t>(t.dr),
                    C(0.0, 0.0));
                const int kp = keep_pos[static_cast<std::size_t>(site)];
                for (int lb = 0; lb < t.dl; ++lb)
                {
                    for (int lk = 0; lk < t.dl; ++lk)
                    {
                        const C e = env[static_cast<std::size_t>(lb) *
                                            static_cast<std::size_t>(dim) +
                                        static_cast<std::size_t>(lk)];
                        if (std::abs(e) == 0.0) continue;
                        if (kp >= 0)
                        {
                            const int sb = row_bits[kp];
                            const int sk = col_bits[kp];
                            for (int rb = 0; rb < t.dr; ++rb)
                            {
                                const C vb = t(lb, sb, rb);
                                if (std::abs(vb) == 0.0) continue;
                                for (int rk = 0; rk < t.dr; ++rk)
                                {
                                    next[static_cast<std::size_t>(rb) *
                                             static_cast<std::size_t>(t.dr) +
                                         static_cast<std::size_t>(rk)] +=
                                        e * vb * std::conj(t(lk, sk, rk));
                                }
                            }
                        }
                        else
                        {
                            for (int s = 0; s < 2; ++s)
                            {
                                const double factor =
                                    (use_twist && twist != nullptr &&
                                     twist[static_cast<std::size_t>(site)] &&
                                     s == 1)
                                        ? -1.0
                                        : 1.0;
                                for (int rb = 0; rb < t.dr; ++rb)
                                {
                                    const C vb = t(lb, s, rb);
                                    if (std::abs(vb) == 0.0) continue;
                                    for (int rk = 0; rk < t.dr; ++rk)
                                    {
                                        next[static_cast<std::size_t>(rb) *
                                                 static_cast<std::size_t>(t.dr) +
                                             static_cast<std::size_t>(rk)] +=
                                            factor * e * vb *
                                            std::conj(t(lk, s, rk));
                                    }
                                }
                            }
                        }
                    }
                }
                env.swap(next);
                dim = t.dr;
            }
            return env.empty() ? C(0.0, 0.0) : env[0];
        }
    };

    inline Gate2 gate_i() { return {C(1, 0), C(0, 0), C(0, 0), C(1, 0)}; }
    inline Gate2 gate_x() { return {C(0, 0), C(1, 0), C(1, 0), C(0, 0)}; }
    inline Gate2 gate_h()
    {
        const double s = 1.0 / std::sqrt(2.0);
        return {C(s, 0), C(s, 0), C(s, 0), C(-s, 0)};
    }
    inline Gate2 gate_s() { return {C(1, 0), C(0, 0), C(0, 0), C(0, 1)}; }
    inline Gate2 gate_t()
    {
        const double a = M_PI / 4.0;
        return {C(1, 0), C(0, 0), C(0, 0), std::exp(C(0, a))};
    }
    inline Gate2 gate_t_adj()
    {
        const double a = -M_PI / 4.0;
        return {C(1, 0), C(0, 0), C(0, 0), std::exp(C(0, a))};
    }
    inline Gate2 gate_rz(double theta)
    {
        return {std::exp(C(0, -0.5 * theta)), C(0, 0), C(0, 0), std::exp(C(0, 0.5 * theta))};
    }
    inline Gate2 gate_rx(double theta)
    {
        const double c = std::cos(0.5 * theta);
        const C s(0.0, -std::sin(0.5 * theta));
        return {C(c, 0), s, s, C(c, 0)};
    }
    inline Gate2 gate_ry(double theta)
    {
        const double c = std::cos(0.5 * theta);
        const double s = std::sin(0.5 * theta);
        return {C(c, 0), C(-s, 0), C(s, 0), C(c, 0)};
    }
    inline Gate2 matmul(const Gate2 &a, const Gate2 &b)
    {
        return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
                a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]};
    }

    inline Gate4 identity4()
    {
        Gate4 u{};
        u[0] = u[5] = u[10] = u[15] = C(1, 0);
        return u;
    }

    inline Gate4 swap_gate()
    {
        Gate4 u{};
        u[0] = C(1, 0);
        u[6] = C(1, 0);
        u[9] = C(1, 0);
        u[15] = C(1, 0);
        return u;
    }

    inline Gate4 cz_gate()
    {
        Gate4 u = identity4();
        u[15] = C(-1, 0);
        return u;
    }

    inline Gate4 cnot_gate()
    {
        // control is first/local low bit q0, target is second bit q1.
        Gate4 u{};
        u[0] = C(1, 0);  // 00 -> 00
        u[1 * 4 + 3] = C(1, 0); // 11 -> 01? wait row=01 col=11
        u[2 * 4 + 2] = C(1, 0); // 10 -> 10
        u[3 * 4 + 1] = C(1, 0); // 01 -> 11
        return u;
    }

    inline Gate4 rxx_gate(double theta)
    {
        // exp(-i theta/2 X X), local basis q0 + 2*q1.
        const double c = std::cos(0.5 * theta);
        const C s(0.0, -std::sin(0.5 * theta));
        Gate4 u{};
        u[0] = u[5] = u[10] = u[15] = C(c, 0);
        u[3] = s;   // col 3 -> row 0
        u[6] = s;   // col 2 -> row 1
        u[9] = s;   // col 1 -> row 2
        u[12] = s;  // col 0 -> row 3
        return u;
    }

    inline Gate4 rzz_gate_exp_plus(double alpha)
    {
        // exp(+i alpha Z Z), used for R_ZZ(pi/4)=exp(+i*pi/8 ZZ).
        Gate4 u{};
        const C same = std::exp(C(0.0, alpha));
        const C diff = std::exp(C(0.0, -alpha));
        u[0] = same;
        u[5] = diff;
        u[10] = diff;
        u[15] = same;
        return u;
    }

    inline Gate4 fswap_gate()
    {
        Gate4 u = swap_gate();
        u[15] = C(-1, 0);
        return u;
    }

    inline Gate2 u3_matrix(const U2Params &p)
    {
        const double c = std::cos(0.5 * p.theta);
        const double s = std::sin(0.5 * p.theta);
        const C phase = std::exp(C(0.0, p.global_phase));
        const C ephi = std::exp(C(0.0, p.phi));
        const C elam = std::exp(C(0.0, p.lambda));
        return {phase * C(c, 0.0), phase * (-elam * C(s, 0.0)),
                phase * (ephi * C(s, 0.0)), phase * (ephi * elam * C(c, 0.0))};
    }

    inline Gate4 parity_preserving_gate_matrix(const RppuBondGate &gate)
    {
        const Gate2 even = u3_matrix(gate.even);
        const Gate2 odd = u3_matrix(gate.odd);
        Gate4 u{};
        // even block on |00>, |11>
        u[0 * 4 + 0] = even[0];
        u[0 * 4 + 3] = even[1];
        u[3 * 4 + 0] = even[2];
        u[3 * 4 + 3] = even[3];
        // odd block on |01>, |10>
        u[1 * 4 + 1] = odd[0];
        u[1 * 4 + 2] = odd[1];
        u[2 * 4 + 1] = odd[2];
        u[2 * 4 + 2] = odd[3];
        return u;
    }

    inline Gate4 haar4_to_gate(const Haar4 &h)
    {
        Gate4 u{};
        for (std::size_t i = 0; i < 16; ++i) u[i] = h[i];
        return u;
    }

    inline void apply_mms_bond(MPS &mps, int i, int j, const MmsLayer &layer)
    {
        auto apply_local_choice = [&](int q) {
            if (layer.rot_x_flags[static_cast<std::size_t>(q)]) mps.apply_one_site(q, gate_rx(M_PI / 2.0));
            if (layer.rot_y_flags[static_cast<std::size_t>(q)]) mps.apply_one_site(q, gate_ry(M_PI / 2.0));
            if (layer.rot_xy_flags[static_cast<std::size_t>(q)])
            {
                mps.apply_one_site(q, gate_rz(-M_PI / 4.0));
                mps.apply_one_site(q, gate_rx(M_PI / 2.0));
                mps.apply_one_site(q, gate_rz(M_PI / 4.0));
            }
        };
        apply_local_choice(i);
        apply_local_choice(j);
        mps.apply_two_site(i, j, rxx_gate(M_PI / 2.0));
    }

    inline void simulate_mms(MPS &mps, int n, const std::vector<MmsLayer> &layers, std::mt19937 &rng)
    {
        int gate_counter = 0;
        const int norm_every = normalize_interval();
        for (const auto &layer : layers)
        {
            for (int i = layer.start; i < n - 1; i += 2)
            {
                apply_mms_bond(mps, i, i + 1, layer);
                if (++gate_counter % norm_every == 0) mps.normalize();
            }
            if (layer.start == 1 && n > 2)
            {
                apply_mms_bond(mps, n - 1, 0, layer);
                if (++gate_counter % norm_every == 0) mps.normalize();
            }
            mps.measure_layer(layer.measure_flags, rng);
        }
    }

    inline void simulate_haar(MPS &mps, int n, int periods, double p, std::mt19937 &rng)
    {
        std::bernoulli_distribution measure_dist(p);
        int gate_counter = 0;
        const int norm_every = normalize_interval();
        auto meas = [&]() {
            std::vector<int> flags(static_cast<std::size_t>(n), 0);
            for (int q = 0; q < n; ++q) flags[static_cast<std::size_t>(q)] = measure_dist(rng) ? 1 : 0;
            mps.measure_layer(flags, rng);
        };
        auto layer = [&](int start) {
            for (int q = start; q < n - 1; q += 2)
            {
                mps.apply_two_site(q, q + 1, haar4_to_gate(haar_unitary_4(rng)));
                if (++gate_counter % norm_every == 0) mps.normalize();
            }
            if (start == 1 && n > 2)
            {
                mps.apply_two_site(0, n - 1, haar4_to_gate(haar_unitary_4(rng)));
                if (++gate_counter % norm_every == 0) mps.normalize();
            }
        };
        const int layer_limit = prefix_layers();
        int layers_done = 0;
        for (int period = 0; period < periods; ++period)
        {
            if (layer_limit > 0 && layers_done >= layer_limit) break;
            layer(0); meas(); ++layers_done;
            if (layer_limit > 0 && layers_done >= layer_limit) break;
            layer(1); meas(); ++layers_done;
        }
    }

    inline void simulate_fermion(MPS &mps, const std::vector<RppuLayer> &layers, std::mt19937 &rng)
    {
        int gate_counter = 0;
        const int norm_every = normalize_interval();
        for (const auto &layer : layers)
        {
            for (const auto &gate : layer.gates)
            {
                if (gate.kind == 1)
                {
                    mps.apply_two_site(gate.q0, gate.q1, fswap_gate());
                }
                else if (gate.kind == 2)
                {
                    // Native MPS does not use the nonlocal JW-CZ shortcut; use
                    // an adjacent swap network around the direct two-mode gate.
                    mps.apply_two_site(gate.q0, gate.q1, parity_preserving_gate_matrix(gate));
                }
                else
                {
                    mps.apply_two_site(gate.q0, gate.q1, parity_preserving_gate_matrix(gate));
                }
                if (++gate_counter % norm_every == 0) mps.normalize();
            }
            mps.measure_layer(layer.measure_flags, rng);
        }
    }

    inline void simulate_rfgs(MPS &mps, int n, const std::vector<RfgsLayer> &layers, bool closed, std::mt19937 &rng)
    {
        (void)rng;
        int gate_counter = 0;
        const int norm_every = normalize_interval();
        for (const auto &layer : layers)
        {
            const int bond_stop = (closed && layer.start == 1 && n > 2) ? n : (n - 1);
            int bond_index = 0;
            for (int i = layer.start; i < bond_stop; i += 2)
            {
                const int j = (i + 1) % n;
                const int local_i = layer.local_left_mask[static_cast<std::size_t>(bond_index)];
                const int local_j = layer.local_right_mask[static_cast<std::size_t>(bond_index)];
                const bool wrapping = (j < i);
                if (wrapping)
                {
                    for (int k = j + 1; k < i; ++k) mps.apply_two_site(k, i, cz_gate());
                    // Same local-gate/S^dagger fusion as the CUDA-Q RFGS kernel.
                    if (local_i == 1) mps.apply_one_site(i, gate_t_adj());
                    if (local_j == 1) mps.apply_one_site(j, gate_t_adj());
                    mps.apply_two_site(i, j, rxx_gate(-M_PI / 2.0));
                    mps.apply_one_site(i, gate_s());
                    mps.apply_one_site(j, gate_s());
                    mps.apply_two_site(i, j, rzz_gate_exp_plus(M_PI / 8.0));
                    for (int k = j + 1; k < i; ++k) mps.apply_two_site(k, i, cz_gate());
                }
                else
                {
                    if (local_i == 1) mps.apply_one_site(i, gate_t()); else mps.apply_one_site(i, gate_s());
                    if (local_j == 1) mps.apply_one_site(j, gate_t()); else mps.apply_one_site(j, gate_s());
                    mps.apply_two_site(i, j, rxx_gate(-M_PI / 2.0));
                    mps.apply_two_site(i, j, rzz_gate_exp_plus(M_PI / 8.0));
                }
                if (++gate_counter % norm_every == 0) mps.normalize();
                ++bond_index;
            }
            mps.measure_layer(layer.measure_flags, rng);
        }
    }

    inline void simulate_1d_rho3(int n,
                                 int periods,
                                 double p,
                                 CircuitType circ_type,
                                 const std::vector<Subsystem> &subsystems,
                                 double *rho_ri,
                                 std::mt19937 &rng)
    {
        const int chi = max_bond();
        MPS mps(n, chi, abs_cutoff(), rel_cutoff());
        if (verbose())
        {
            log("start: n=" + std::to_string(n) + " periods=" + std::to_string(periods) +
                " p=" + std::to_string(p) + " circ_type=" + std::to_string(static_cast<int>(circ_type)) +
                " max_bond=" + std::to_string(chi) + " abs_cutoff=" + std::to_string(abs_cutoff()) +
                " rel_cutoff=" + std::to_string(rel_cutoff()));
        }

        if (circ_type == CircuitType::MMS)
        {
            std::vector<MmsLayer> layers;
            build_mms_layers(layers, n, periods, p, rng);
            apply_debug_prefix_layer_limit(layers, "native-MPS MMS");
            simulate_mms(mps, n, layers, rng);
        }
        else if (circ_type == CircuitType::Haar)
        {
            simulate_haar(mps, n, periods, p, rng);
        }
        else if (circ_type == CircuitType::FermionRPPU)
        {
            std::vector<RppuLayer> layers;
            // Use adjacent FSWAP chains in the native MPS path; this avoids
            // nonlocal JW-CZ tensor contractions while preserving the original
            // closed-boundary construction.
            build_rppu_layers(layers, n, periods, p, true, rng, false);
            apply_debug_prefix_layer_limit(layers, "native-MPS FermionRPPU");
            simulate_fermion(mps, layers, rng);
        }
        else if (circ_type == CircuitType::RFGS)
        {
            std::vector<RfgsLayer> layers;
            build_rfgs_layers(layers, n, periods, p, rng, true);
            apply_debug_prefix_layer_limit(layers, "native-MPS RFGS");
            simulate_rfgs(mps, n, layers, true, rng);
        }
        else if (circ_type == CircuitType::QubitRPPU)
        {
            std::vector<RppuLayer> layers;
            build_qrppu_layers(layers, n, periods, p, true, rng);
            apply_debug_prefix_layer_limit(layers, "native-MPS qRPPU");
            // The direct periodic qRPPU bond is transported with ordinary
            // qubit SWAPs by MPS::apply_long_range; no fermionic sign string
            // is inserted. Reduced states are traced as qubits below.
            simulate_fermion(mps, layers, rng);
        }
        else
        {
            throw std::invalid_argument("Unsupported circuit type for native MPS.");
        }

        mps.normalize();
        if (verbose()) log("evolution done; max_observed_bond=" + std::to_string(mps.max_observed_bond()) + "; extracting rho3");
        const bool fermion_trace = uses_fermionic_trace(circ_type);
        for (std::size_t s = 0; s < subsystems.size(); ++s)
        {
            mps.rho3(subsystems[s], rho_ri + s * RHO3_VALUES, fermion_trace);
        }
        if (verbose()) log("rho3 extraction done");
    }
}

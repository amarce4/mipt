// Validates the fused bond matrices (optimization #1) against ground truth
// extracted from the CUDA-Q runtime by dump_bond.cpp.
//
//   ./dump_bond 12345 > ground_truth.txt
//   ./cusv_gate_tests ground_truth.txt
//
// Also checks the structural invariants that make optimization #5 safe:
// blocking only groups consecutive disjoint ops, and the blocked matrix is
// exactly the Kronecker product of the bond matrices it replaces.

#include "mipt/cusv/gates.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using mipt::cusv::Complex;
using mipt::cusv::DenseMatrix;

namespace
{

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::cout << "  FAIL: " << what << "\n";
        ++failures;
    }
}

double max_abs_diff(const DenseMatrix &m, const std::vector<Complex> &ref)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        worst = std::max(worst, std::abs(m.values[i] - ref[i]));
    }
    return worst;
}

// Reads the 4x4 block that follows `label` in the ground-truth dump.
std::vector<Complex> read_matrix(const std::vector<std::string> &lines, const std::string &label)
{
    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        if (lines[i].rfind(label, 0) != 0) { continue; }
        std::vector<Complex> out;
        for (int r = 1; r <= 4; ++r)
        {
            std::istringstream in(lines[i + static_cast<std::size_t>(r)]);
            for (int c = 0; c < 4; ++c)
            {
                double re = 0.0;
                double im = 0.0;
                in >> re >> im;
                out.emplace_back(re, im);
            }
        }
        return out;
    }
    std::cout << "  FAIL: label not found in ground truth: " << label << "\n";
    ++failures;
    return {};
}

void test_against_cudaq(const std::vector<std::string> &lines, unsigned seed)
{
    std::mt19937 rng(seed);

    // Same draw order as dump_bond.cpp, so the sampled gates are identical.
    {
        const auto gate = mipt::random_parity_preserving_gate(0, 1, rng);
        const auto ref = read_matrix(lines, "RPPU_MATRIX");
        const auto mine = mipt::cusv::rppu_bond_matrix(gate);
        const double d = ref.empty() ? 1.0 : max_abs_diff(mine, ref);
        std::printf("  RPPU fused bond vs CUDA-Q 8-gate decomposition: max|diff| = %.3e\n", d);
        check(d < 1e-13, "RPPU bond matrix");
    }

    {
        const auto u = mipt::haar_unitary_4(rng);
        const auto gate = mipt::decompose_haar_u4_csd(0, 1, u);
        const auto ref = read_matrix(lines, "HAAR_MATRIX");
        const auto mine = mipt::cusv::haar_bond_matrix(gate);
        const double d = ref.empty() ? 1.0 : max_abs_diff(mine, ref);
        std::printf("  Haar fused bond vs CUDA-Q 20-gate CSD:          max|diff| = %.3e\n", d);
        check(d < 1e-13, "Haar bond matrix");

        // The fused matrix should also reproduce the originally sampled U(4).
        const auto sampled = read_matrix(lines, "HAAR_SAMPLED");
        if (!sampled.empty())
        {
            const double ds = max_abs_diff(mine, sampled);
            std::printf("  Haar fused bond vs sampled U(4):                max|diff| = %.3e\n", ds);
            check(ds < 1e-12, "Haar bond reproduces sampled U(4)");
        }
    }

    {
        double worst = 0.0;
        for (int a0 = 0; a0 < 3; ++a0)
        {
            for (int a1 = 0; a1 < 3; ++a1)
            {
                char label[64];
                std::snprintf(label, sizeof(label), "MMS_MATRIX axis0=%d axis1=%d", a0, a1);
                const auto ref = read_matrix(lines, label);
                if (ref.empty()) { continue; }
                const auto mine = mipt::cusv::mms_bond_matrix(a0, a1);
                worst = std::max(worst, max_abs_diff(mine, ref));
            }
        }
        std::printf("  MMS fused bond vs CUDA-Q (all 9 axis pairs):    max|diff| = %.3e\n", worst);
        check(worst < 1e-13, "MMS bond matrices");
    }
}

void test_unitarity()
{
    std::mt19937 rng(999);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial)
    {
        const auto rppu = mipt::cusv::rppu_bond_matrix(mipt::random_parity_preserving_gate(0, 1, rng));
        const auto haar = mipt::cusv::haar_bond_matrix(
            mipt::decompose_haar_u4_csd(0, 1, mipt::haar_unitary_4(rng)));
        for (const auto *m : {&rppu, &haar})
        {
            for (std::size_t i = 0; i < 4; ++i)
            {
                for (std::size_t j = 0; j < 4; ++j)
                {
                    Complex acc{0.0, 0.0};
                    for (std::size_t k = 0; k < 4; ++k)
                    {
                        acc += m->at(i, k) * std::conj(m->at(j, k));
                    }
                    const Complex expected = (i == j) ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
                    worst = std::max(worst, std::abs(acc - expected));
                }
            }
        }
    }
    std::printf("  unitarity over 400 random bonds:               max|UU*-I| = %.3e\n", worst);
    check(worst < 1e-12, "bond matrices are unitary");
}

void test_parity_block_structure()
{
    std::mt19937 rng(4242);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial)
    {
        const auto m = mipt::cusv::rppu_bond_matrix(mipt::random_parity_preserving_gate(0, 1, rng));
        // Off-block entries connect different computational parities.
        for (std::size_t r = 0; r < 4; ++r)
        {
            for (std::size_t c = 0; c < 4; ++c)
            {
                const int pr = (static_cast<int>(r) & 1) ^ ((static_cast<int>(r) >> 1) & 1);
                const int pc = (static_cast<int>(c) & 1) ^ ((static_cast<int>(c) >> 1) & 1);
                if (pr != pc)
                {
                    worst = std::max(worst, std::abs(m.at(r, c)));
                }
            }
        }
    }
    std::printf("  RPPU parity conservation:                 max|off-block| = %.3e\n", worst);
    check(worst == 0.0, "RPPU bond is exactly parity block diagonal");
}

// #5: the blocked matrix must equal the Kronecker product of what it replaced,
// and blocking must never merge across a JwString or overlapping supports.
void test_blocking()
{
    std::mt19937 rng(7);
    std::vector<mipt::cusv::Op> ops;
    for (int b = 0; b < 4; ++b)
    {
        mipt::cusv::Op op;
        op.targets = {2 * b, 2 * b + 1};
        op.matrix = mipt::cusv::rppu_bond_matrix(mipt::random_parity_preserving_gate(2 * b, 2 * b + 1, rng));
        ops.push_back(std::move(op));
    }

    const auto blocked = mipt::cusv::block_disjoint_ops(ops, 6);
    check(blocked.size() == 2, "4 disjoint bonds at max=6 targets give 2 blocks");
    check(blocked[0].matrix.targets == 6, "first block covers 3 bonds (6 targets)");
    if (!blocked.empty())
    {
        std::printf("  blocking 4 disjoint bonds (max 6 targets): %zu ops -> %zu ops (%d, %d targets)\n",
                    ops.size(), blocked.size(), blocked[0].matrix.targets,
                    blocked.size() > 1 ? blocked[1].matrix.targets : 0);
    }

    // Entry-by-entry Kronecker check on the first block.
    if (!blocked.empty() && blocked[0].matrix.targets == 6)
    {
        double worst = 0.0;
        const auto &B = blocked[0].matrix;
        for (std::size_t r = 0; r < 64; ++r)
        {
            for (std::size_t c = 0; c < 64; ++c)
            {
                const Complex expected = ops[0].matrix.at(r & 3u, c & 3u) *
                                         ops[1].matrix.at((r >> 2) & 3u, (c >> 2) & 3u) *
                                         ops[2].matrix.at((r >> 4) & 3u, (c >> 4) & 3u);
                worst = std::max(worst, std::abs(B.at(r, c) - expected));
            }
        }
        std::printf("  block == Kronecker product of its bonds:        max|diff| = %.3e\n", worst);
        check(worst < 1e-15, "blocked matrix factorizes exactly");
        check(blocked[0].targets == std::vector<int>({0, 1, 2, 3, 4, 5}), "block target order");
    }

    // A JwString between two bonds must act as a barrier.
    std::vector<mipt::cusv::Op> with_string;
    with_string.push_back(ops[0]);
    mipt::cusv::Op s;
    s.kind = mipt::cusv::OpKind::JwString;
    s.string_control = 7;
    s.string_mask = 0b0110;
    with_string.push_back(s);
    with_string.push_back(ops[1]);
    const auto barrier = mipt::cusv::block_disjoint_ops(with_string, 6);
    check(barrier.size() == 3, "blocking does not merge across a JwString op");

    // Overlapping supports must not merge either.
    std::vector<mipt::cusv::Op> overlap;
    overlap.push_back(ops[0]);
    mipt::cusv::Op shared = ops[1];
    shared.targets = {1, 2};
    overlap.push_back(shared);
    const auto kept = mipt::cusv::block_disjoint_ops(overlap, 6);
    check(kept.size() == 2, "blocking does not merge ops with overlapping targets");
}

} // namespace

int main(int argc, char **argv)
{
    std::cout << "cusv_gate_tests\n";

    if (argc > 1)
    {
        std::ifstream in(argv[1]);
        if (!in)
        {
            std::cout << "could not open ground truth file " << argv[1] << "\n";
            return 1;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) { lines.push_back(line); }
        const unsigned seed = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 12345u;
        test_against_cudaq(lines, seed);
    }
    else
    {
        std::cout << "  (no ground-truth file given; skipping CUDA-Q comparison)\n";
    }

    test_unitarity();
    test_parity_block_structure();
    test_blocking();

    if (failures == 0)
    {
        std::cout << "cusv_gate_tests: PASS\n";
        return 0;
    }
    std::cout << "cusv_gate_tests: " << failures << " FAILURE(S)\n";
    return 1;
}

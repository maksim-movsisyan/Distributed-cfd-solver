// Distributed linear algebra module tests: ghost exchange, fused reductions,
// CSR/BSR COO assembly (+ fixed-pattern value reassembly), and distributed
// BiCGSTAB with SGS / identity preconditioning on 2D Poisson problems.
//
// Run via ctest (1..4 ranks) or directly: mpirun -np 4 cfd_linalg_test.
// Set LINALG_PERF=1 in the environment for a performance smoke report.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "cfd/linalg/linalg.hpp"

namespace {

using namespace cfd::linalg;
using cfd::GlobalIndex;
using cfd::LocalIndex;
using cfd::kInvalidLocalIndex;

constexpr double kPi = 3.14159265358979323846;

int g_rank = 0;
int g_nprocs = 1;

void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "[r%d] FAILED: %s\n", g_rank, what);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void pass(const char* name) {
    if (g_rank == 0) std::printf("[linalg] %-46s ok\n", name);
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// Same balanced contiguous split as VectorLayout's internal prefix sum.
LocalIndex balanced_local(GlobalIndex n_global) {
    const GlobalIndex p = static_cast<GlobalIndex>(g_nprocs);
    return static_cast<LocalIndex>(n_global / p + ((static_cast<GlobalIndex>(g_rank) < n_global % p) ? 1 : 0));
}

// --- ghost exchange + vector kernels -----------------------------------------

void test_ghosts_and_reductions() {
    const LocalIndex per = 5;
    const GlobalIndex ng = static_cast<GlobalIndex>(per) * g_nprocs;
    VectorLayout layout(MPI_COMM_WORLD, ng, per);

    std::vector<GlobalIndex> ghosts;
    if (g_nprocs > 1) {
        const GlobalIndex base = static_cast<GlobalIndex>(per) * ((g_rank + 1) % g_nprocs);
        ghosts = {base, base + 2};
    }
    layout.setGhosts(ghosts);  // collective
    require(layout.ghostSize() == static_cast<LocalIndex>(ghosts.size()), "ghost count");
    require(layout.globalSize() == ng, "global size");

    const GlobalIndex begin = layout.localBegin();
    require(layout.localIndex(begin) == 0, "localIndex owned");
    require(layout.localIndex(begin + per - 1) == per - 1, "localIndex owned last");
    if (g_nprocs > 1) {
        require(layout.localIndex(ghosts[0]) == per, "localIndex first ghost");
        require(layout.localIndex(ghosts[1]) == per + 1, "localIndex second ghost");
    }
    require(layout.localIndex(ng + 100) == kInvalidLocalIndex, "localIndex invalid");

    // Scalar ghost exchange.
    Vector v(layout, 1);
    for (LocalIndex i = 0; i < per; ++i) v.owned()[i] = static_cast<double>(begin + i);
    v.updateGhosts();
    for (LocalIndex k = 0; k < layout.ghostSize(); ++k) {
        require(v.ghosts()[k] == static_cast<double>(ghosts[k]), "scalar ghost value");
    }

    // Block ghost exchange.
    Vector w(layout, 3);
    for (LocalIndex i = 0; i < per; ++i) {
        for (int c = 0; c < 3; ++c) w.owned()[static_cast<std::size_t>(i) * 3 + c] = static_cast<double>(begin + i) * 10.0 + c;
    }
    w.updateGhosts();
    for (LocalIndex k = 0; k < layout.ghostSize(); ++k) {
        for (int c = 0; c < 3; ++c) {
            require(w.ghosts()[static_cast<std::size_t>(k) * 3 + c] == static_cast<double>(ghosts[k]) * 10.0 + c,
                    "block ghost value");
        }
    }

    // Reductions: v holds the global id per row, so sums are known exactly.
    double s2 = 0.0, s1 = 0.0;
    for (GlobalIndex g = 0; g < ng; ++g) {
        s2 += static_cast<double>(g) * g;
        s1 += static_cast<double>(g);
    }
    require(near(v.dot(v), s2, 1e-12), "dot global");
    require(near(v.norm2(), std::sqrt(s2), 1e-12), "norm2 global");

    Vector ones(layout, 1);
    for (LocalIndex i = 0; i < per; ++i) ones.owned()[i] = 1.0;
    require(near(v.dot(ones), s1, 1e-12), "dot ones");

    double out[2];
    const std::pair<const Vector*, const Vector*> prods[] = {{&v, &v}, {&v, &ones}};
    Vector::batchedDots(prods, out);
    require(near(out[0], s2, 1e-12) && near(out[1], s1, 1e-12), "batchedDots");

    // Element-wise kernels.
    Vector c1(layout, 1);
    c1.copyFrom(v);
    c1.scale(2.0);
    c1.axpy(1.0, v);
    for (LocalIndex i = 0; i < per; ++i) {
        require(c1.owned()[i] == 3.0 * v.owned()[i], "scale/axpy");
    }
}

// --- COO duplicate merging + values-only reassembly ---------------------------

void test_assembly_semantics() {
    const LocalIndex per = 4;
    const GlobalIndex ng = static_cast<GlobalIndex>(per) * g_nprocs;

    // CSR: diag 2.0 (as 1+1), sub 1.0 (as 0.5+0.25+0.25), super 0.5.
    CsrMatrix A(MPI_COMM_WORLD, ng, per);
    const GlobalIndex begin = A.layout().localBegin();
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        A.addValue(g, g, 1.0);
        A.addValue(g, g, 1.0);
        if (g > 0) {
            A.addValue(g, g - 1, 0.5);
            A.addValue(g, g - 1, 0.25);
            A.addValue(g, g - 1, 0.25);
        }
        if (g + 1 < ng) A.addValue(g, g + 1, 0.5);
    }
    A.assemble();

    Vector x = A.makeVector();
    for (LocalIndex i = 0; i < per; ++i) x.owned()[i] = static_cast<double>(begin + i);
    x.updateGhosts();
    Vector y = A.makeVector();
    A.apply(x, y);

    auto expected_csr = [&](GlobalIndex g) {
        double e = 2.0 * g;
        if (g > 0) e += 1.0 * (g - 1);
        if (g + 1 < ng) e += 0.5 * (g + 1);
        return e;
    };
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        require(near(y.owned()[i], expected_csr(g), 1e-12), "csr apply after COO merge");
        require(A.diagIndex()[i] != kInvalidLocalIndex, "csr diag slot present");
    }

    // Values-only reassembly: add 0.5 to the diagonal.
    for (LocalIndex i = 0; i < per; ++i) A.addValue(begin + i, begin + i, 0.5);
    A.assembleValues();
    A.apply(x, y);
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        require(near(y.owned()[i], expected_csr(g) + 0.5 * g, 1e-12), "csr apply after reassembly");
    }

    // BSR (bs = 2) with a non-symmetric diagonal block.
    BsrMatrix B(MPI_COMM_WORLD, ng, per, 2);
    const double dblock[4] = {2.0, 0.1, -0.05, 2.0};
    const double iblock[4] = {0.25, 0.0, 0.0, 0.25};
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        B.addBlock(g, g, dblock);
        if (g > 0) {
            B.addBlock(g, g - 1, iblock);
            B.addBlock(g, g - 1, iblock);  // -> 0.5 I
        }
        if (g + 1 < ng) B.addBlock(g, g + 1, iblock);
    }
    B.assemble();

    Vector xb = B.makeVector();
    for (LocalIndex i = 0; i < per; ++i) {
        for (int k = 0; k < 2; ++k) {
            xb.owned()[static_cast<std::size_t>(i) * 2 + k] = static_cast<double>(begin + i) * 3.0 + k;
        }
    }
    xb.updateGhosts();
    Vector yb = B.makeVector();
    B.apply(xb, yb);

    auto xg = [&](GlobalIndex g, int k) { return static_cast<double>(g) * 3.0 + k; };
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        double e0 = 2.0 * xg(g, 0) + 0.1 * xg(g, 1);
        double e1 = -0.05 * xg(g, 0) + 2.0 * xg(g, 1);
        if (g > 0) {
            e0 += 0.5 * xg(g - 1, 0);
            e1 += 0.5 * xg(g - 1, 1);
        }
        if (g + 1 < ng) {
            e0 += 0.25 * xg(g + 1, 0);
            e1 += 0.25 * xg(g + 1, 1);
        }
        require(near(yb.owned()[static_cast<std::size_t>(i) * 2], e0, 1e-12) &&
                    near(yb.owned()[static_cast<std::size_t>(i) * 2 + 1], e1, 1e-12),
                "bsr apply after COO merge");
    }

    const double half[4] = {0.5, 0.0, 0.0, 0.5};
    for (LocalIndex i = 0; i < per; ++i) B.addBlock(begin + i, begin + i, half);
    B.assembleValues();
    B.apply(xb, yb);
    for (LocalIndex i = 0; i < per; ++i) {
        const GlobalIndex g = begin + i;
        require(near(yb.owned()[static_cast<std::size_t>(i) * 2 + 1],
                     2.0 * xg(g, 1) + 0.5 * xg(g, 1) - 0.05 * xg(g, 0) +
                         (g > 0 ? 0.5 * xg(g - 1, 1) : 0.0) +
                         (g + 1 < ng ? 0.25 * xg(g + 1, 1) : 0.0),
                     1e-12),
                "bsr apply after reassembly");
    }
}

// --- distributed BiCGSTAB on 2D Poisson ----------------------------------------

// Mixed-mode exact solution (NOT a single eigenmode of the Poisson operator:
// a pure fundamental mode would make unpreconditioned Krylov methods converge
// in one iteration and hide solver defects).
double poisson_exact(GlobalIndex g, int n) {
    const int gi = static_cast<int>(g % n), gj = static_cast<int>(g / n);
    const double xi = static_cast<double>(gi + 1) / (n + 1.0);
    const double yj = static_cast<double>(gj + 1) / (n + 1.0);
    return std::sin(kPi * xi) * std::sin(kPi * yj) +
           0.3 * std::sin(3.0 * kPi * xi) * std::sin(2.0 * kPi * yj);
}

// Fills the solution vector with the exact field and returns b = A x_exact.
Vector poisson_rhs(const LinearOperator& A, const Vector& xt) {
    xt.updateGhosts();
    Vector b = A.makeVector();
    A.apply(xt, b);
    return b;
}

void test_csr_poisson() {
    // SGS-preconditioned.
    {
        const int n = 41;
        const GlobalIndex ng = static_cast<GlobalIndex>(n) * n;
        CsrMatrix A(MPI_COMM_WORLD, ng, balanced_local(ng));
        const double c = static_cast<double>(n + 1) * (n + 1);
        const GlobalIndex begin = A.layout().localBegin();
        for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
            const GlobalIndex g = begin + i;
            const int gi = static_cast<int>(g % n), gj = static_cast<int>(g / n);
            A.addValue(g, g, 4.0 * c);
            if (gi > 0) A.addValue(g, g - 1, -c);
            if (gi + 1 < n) A.addValue(g, g + 1, -c);
            if (gj > 0) A.addValue(g, g - n, -c);
            if (gj + 1 < n) A.addValue(g, g + n, -c);
        }
        A.assemble();

        Vector xt = A.makeVector();
        for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
            xt.owned()[i] = poisson_exact(begin + i, n);
        }
        Vector b = poisson_rhs(A, xt);

        Vector x = A.makeVector();
        SgsPreconditioner sgs;
        sgs.setup(A);
        BiCGSTAB solver;
        solver.params().relative_tolerance = 1e-10;
        solver.params().max_iterations = 2000;
        solver.params().verbosity = Verbosity::Silent;
        const IterationResult res = solver.solve(A, sgs, x, b);
        if (g_rank == 0) std::printf("    [info] csr poisson sgs: %d iterations\n", res.iterations);
        require(res.status == SolverStatus::Converged, "csr poisson sgs: converged");
        require(res.iterations < 4000, "csr poisson sgs: iteration count sane");
        require(near(res.true_final_residual / b.norm2(), 0.0, 1e-8), "csr poisson sgs: true residual");

        Vector err = A.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, xt);
        require(err.norm2() <= 1e-6 * xt.norm2(), "csr poisson sgs: solution error");
    }

    // Unpreconditioned (identity), smaller grid: exercises the generic path.
    {
        const int n = 21;
        const GlobalIndex ng = static_cast<GlobalIndex>(n) * n;
        CsrMatrix A(MPI_COMM_WORLD, ng, balanced_local(ng));
        const double c = static_cast<double>(n + 1) * (n + 1);
        const GlobalIndex begin = A.layout().localBegin();
        for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
            const GlobalIndex g = begin + i;
            const int gi = static_cast<int>(g % n), gj = static_cast<int>(g / n);
            A.addValue(g, g, 4.0 * c);
            if (gi > 0) A.addValue(g, g - 1, -c);
            if (gi + 1 < n) A.addValue(g, g + 1, -c);
            if (gj > 0) A.addValue(g, g - n, -c);
            if (gj + 1 < n) A.addValue(g, g + n, -c);
        }
        A.assemble();

        Vector xt = A.makeVector();
        for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
            xt.owned()[i] = poisson_exact(begin + i, n);
        }
        Vector b = poisson_rhs(A, xt);

        Vector x = A.makeVector();
        IdentityPreconditioner idn;
        idn.setup(A);
        BiCGSTAB solver;
        solver.params().relative_tolerance = 1e-8;
        solver.params().max_iterations = 4000;
        solver.params().verbosity = Verbosity::Silent;
        const IterationResult res = solver.solve(A, idn, x, b);
        require(res.status == SolverStatus::Converged, "csr poisson identity: converged");

        Vector err = A.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, xt);
        require(err.norm2() <= 1e-4 * xt.norm2(), "csr poisson identity: solution error");
    }
}

// Assembles a 5-point 2D Poisson block operator with mildly non-symmetric
// diagonal / coupling blocks (so BiCGSTAB, not CG, is genuinely exercised).
template <int BS>
void assemble_poisson_bsr(int n, BsrMatrix& A) {
    const double c = static_cast<double>(n + 1) * (n + 1);
    double dblock[BS * BS > 0 ? BS * BS : 1];
    double oblock[BS * BS > 0 ? BS * BS : 1];
    for (int a = 0; a < BS; ++a) {
        for (int b = 0; b < BS; ++b) {
            const double dp = static_cast<double>(((a * 7 + b * 13) % 11) - 5);
            const double op = static_cast<double>(((a * 5 + b * 3) % 7) - 3);
            dblock[a * BS + b] = (a == b ? 4.0 * c : 0.0) + 0.003 * c * dp / 5.0;
            oblock[a * BS + b] = (a == b ? -1.0 * c : 0.0) + 0.002 * c * op / 3.0;
        }
    }
    const GlobalIndex begin = A.layout().localBegin();
    for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
        const GlobalIndex g = begin + i;
        const int gi = static_cast<int>(g % n), gj = static_cast<int>(g / n);
        A.addBlock(g, g, dblock);
        if (gi > 0) A.addBlock(g, g - 1, oblock);
        if (gi + 1 < n) A.addBlock(g, g + 1, oblock);
        if (gj > 0) A.addBlock(g, g - n, oblock);
        if (gj + 1 < n) A.addBlock(g, g + n, oblock);
    }
    A.assemble();
}

void solve_bsr_and_check(int n, int bs, double rtol, double err_tol, int max_iter) {
    const GlobalIndex ng = static_cast<GlobalIndex>(n) * n;
    BsrMatrix A(MPI_COMM_WORLD, ng, balanced_local(ng), bs);
    // The assembler needs a compile-time block size for its stack buffers; the
    // runtime matrix block size drives the actual kernel dispatch.
    switch (bs) {
        case 1: assemble_poisson_bsr<1>(n, A); break;
        case 5: assemble_poisson_bsr<5>(n, A); break;
        case 9: assemble_poisson_bsr<9>(n, A); break;
        default: require(false, "solve_bsr_and_check: unsupported bs in test"); return;
    }

    Vector xt = A.makeVector();
    const GlobalIndex begin = A.layout().localBegin();
    for (LocalIndex i = 0; i < A.layout().localSize(); ++i) {
        for (int k = 0; k < bs; ++k) {
            xt.owned()[static_cast<std::size_t>(i) * bs + k] =
                poisson_exact(begin + i, n) * (1.0 + 0.1 * k);
        }
    }
    Vector b = poisson_rhs(A, xt);

    Vector x = A.makeVector();
    SgsPreconditioner sgs;
    sgs.setup(A);
    BiCGSTAB solver;
    solver.params().relative_tolerance = rtol;
    solver.params().max_iterations = max_iter;
    solver.params().verbosity = Verbosity::Silent;
    const IterationResult res = solver.solve(A, sgs, x, b);
    // NOTE: any ratio involving norm2()/dot() must be computed OUTSIDE rank
    // guards — they are collectives and would desynchronize the ranks.
    const double final_rel = res.final_residual / b.norm2();
    if (g_rank == 0) {
        std::printf("    [info] bsr poisson sgs (bs=%d): %d iterations, rel %.3e\n", bs,
                    res.iterations, final_rel);
    }
    require(res.status == SolverStatus::Converged, "bsr poisson sgs: converged");
    require(near(res.true_final_residual / b.norm2(), 0.0, 10.0 * rtol),
            "bsr poisson sgs: true residual");

    Vector err = A.makeVector();
    err.copyFrom(x);
    err.axpy(-1.0, xt);
    require(err.norm2() <= err_tol * xt.norm2(), "bsr poisson sgs: solution error");
}

void test_bsr_poisson() {
    solve_bsr_and_check(29, 5, 1e-10, 1e-6, 2500);  // the CFD block size
    solve_bsr_and_check(13, 1, 1e-9, 1e-5, 2000);   // scalar blocks (bs=1 kernel)
    solve_bsr_and_check(7, 9, 1e-9, 1e-5, 2000);    // dynamic-fallback kernel (bs > 8)
}

// --- optional performance smoke (LINALG_PERF=1) --------------------------------

void perf_smoke() {
    const int n = 120;
    const GlobalIndex ng = static_cast<GlobalIndex>(n) * n;
    const LocalIndex nl = balanced_local(ng);
    const double c = static_cast<double>(n + 1) * (n + 1);

    // BSR, bs = 5: SpMV rate + one SGS-preconditioned solve.
    BsrMatrix A(MPI_COMM_WORLD, ng, nl, 5);
    assemble_poisson_bsr<5>(n, A);
    const std::size_t nnzb = A.values().size() / (5 * 5);
    Vector x = A.makeVector(), y = A.makeVector();
    x.setZero();
    for (LocalIndex i = 0; i < nl; ++i) {
        for (int k = 0; k < 5; ++k) {
            x.owned()[static_cast<std::size_t>(i) * 5 + k] = std::sin(0.1 * (A.layout().localBegin() + i) + k);
        }
    }
    x.updateGhosts();

    const int reps = 200;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int r = 0; r < reps; ++r) A.apply(x, y);
    const double t1 = MPI_Wtime();
    const double gflops = 2.0 * static_cast<double>(nnzb) * 25.0 * reps / (t1 - t0) / 1e9;
    if (g_rank == 0) {
        std::printf("[perf] BSR SpMV  bs=5  n=%d: %.1f GFlop/s (%.2f ms per SpMV)\n", n, gflops,
                    1e3 * (t1 - t0) / reps);
    }

    Vector b = A.makeVector();
    A.apply(x, b);
    Vector sol = A.makeVector();
    SgsPreconditioner sgs;
    sgs.setup(A);
    BiCGSTAB solver;
    // Fixed budget as a rate measurement: unrelaxed point-SGS needs thousands
    // of iterations on this deliberately ill-conditioned grid, which measures
    // sustained kernel + communication throughput, not solver quality.
    solver.params().relative_tolerance = 1e-12;
    solver.params().max_iterations = 500;
    solver.params().verbosity = Verbosity::Silent;
    solver.params().verify_final_residual = false;
    MPI_Barrier(MPI_COMM_WORLD);
    const double s0 = MPI_Wtime();
    const IterationResult res = solver.solve(A, sgs, sol, b);
    const double s1 = MPI_Wtime();
    const double final_rel_perf = res.final_residual / b.norm2();
    if (g_rank == 0) {
        std::printf("[perf] BSR solve bs=5: %.2f ms/iter over %d iters, |r|/|b| %.2e\n",
                    1e3 * (s1 - s0) / res.iterations, res.iterations, final_rel_perf);
    }

    // CSR comparison on the same grid.
    CsrMatrix C(MPI_COMM_WORLD, ng, nl);
    const GlobalIndex begin = C.layout().localBegin();
    for (LocalIndex i = 0; i < nl; ++i) {
        const GlobalIndex g = begin + i;
        const int gi = static_cast<int>(g % n), gj = static_cast<int>(g / n);
        C.addValue(g, g, 4.0 * c);
        if (gi > 0) C.addValue(g, g - 1, -c);
        if (gi + 1 < n) C.addValue(g, g + 1, -c);
        if (gj > 0) C.addValue(g, g - n, -c);
        if (gj + 1 < n) C.addValue(g, g + n, -c);
    }
    C.assemble();
    Vector xc = C.makeVector(), yc = C.makeVector();
    for (LocalIndex i = 0; i < nl; ++i) xc.owned()[i] = std::sin(0.1 * (begin + i));
    xc.updateGhosts();
    MPI_Barrier(MPI_COMM_WORLD);
    const double c0 = MPI_Wtime();
    for (int r = 0; r < reps; ++r) C.apply(xc, yc);
    const double c1 = MPI_Wtime();
    const double nnz = static_cast<double>(C.values().size());
    if (g_rank == 0) {
        std::printf("[perf] CSR SpMV        n=%d: %.1f GFlop/s (%.2f ms per SpMV)\n", n,
                    2.0 * nnz * reps / (c1 - c0) / 1e9, 1e3 * (c1 - c0) / reps);
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    test_ghosts_and_reductions();
    pass("ghost exchange + reductions");
    test_assembly_semantics();
    pass("COO merge + values-only reassembly");
    test_csr_poisson();
    pass("CSR Poisson: BiCGSTAB (SGS / identity)");
    test_bsr_poisson();
    pass("BSR Poisson: BiCGSTAB + SGS (bs 1/5/9)");

    if (std::getenv("LINALG_PERF") != nullptr) perf_smoke();

    if (g_rank == 0) std::printf("[linalg] all tests passed (%d ranks)\n", g_nprocs);
    MPI_Finalize();
    return 0;
}

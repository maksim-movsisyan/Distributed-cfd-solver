#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/validate.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/linalg/vector.hpp"
#include "cfd/linalg/vector_layout.hpp"
#include "cfd/linalg/bsr_matrix.hpp"
#include "cfd/linalg/bsr_helpers.hpp"
#include "cfd/linalg/preconditioners.hpp"
#include "cfd/linalg/bicgstab.hpp"

#include "cfd/io/solver_mesh/hdf5_reader.hpp"
#include "cfd/mpi/log.hpp" 

namespace {

void report_test(const char* name, double computed, double analytical, double tol, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const double diff = std::abs(computed - analytical);
    const bool local_pass = (diff <= tol) && !std::isnan(computed);
    int global_pass_int = 0;
    const int local_pass_int = local_pass ? 1 : 0;
    
    MPI_Allreduce(&local_pass_int, &global_pass_int, 1, MPI_INT, MPI_MIN, comm);
    const bool passed = (global_pass_int == 1);

    if (rank == 0) {
        std::printf("----------------------------------------------------------------------\n");
        std::printf("TEST: %s\n", name);
        std::printf("  Computed   : %22.14e\n", computed);
        std::printf("  Analytical : %22.14e\n", analytical);
        std::printf("  Difference : %22.14e (tolerance: %8.1e)\n", diff, tol);
        if (passed) {
            std::printf("  STATUS     : \033[1;32m[ PASS ]\033[0m\n");
        } else {
            std::printf("  STATUS     : \033[1;31m[ FAILED ]\033[0m\n");
        }
    }
}

std::vector<cfd::GlobalIndex> build_ghost_gids(const cfd::mesh::MeshPart& mp, MPI_Comm comm) {
    const MPI_Datatype mpi_local_type = (sizeof(cfd::LocalIndex) == 8) ? MPI_INT64_T : MPI_INT32_T;

    std::vector<cfd::LocalIndex> all_n_own(static_cast<std::size_t>(mp.nprocs), 0);
    MPI_Allgather(&mp.n_own, 1, mpi_local_type,
                  all_n_own.data(), 1, mpi_local_type, comm);

    std::vector<cfd::GlobalIndex> rank_offsets(static_cast<std::size_t>(mp.nprocs) + 1, 0);
    for (int r = 0; r < mp.nprocs; ++r) {
        const std::size_t ur = static_cast<std::size_t>(r);
        rank_offsets[ur + 1] = rank_offsets[ur] + static_cast<cfd::GlobalIndex>(all_n_own[ur]);
    }

    const int n_nb = mp.n_neighbors();
    std::vector<cfd::LocalIndex> donor_local_ids(mp.recv_ghost_local.size(), 0);
    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(n_nb * 2));

    for (int k = 0; k < n_nb; ++k) {
        const std::size_t uk = static_cast<std::size_t>(k);
        const int neighbor = mp.nb_ranks[uk];
        const cfd::LocalIndex offset = mp.recv_offsets[uk];
        const int count = static_cast<int>(mp.recv_offsets[uk + 1] - offset);
        if (count > 0) {
            MPI_Request req = MPI_REQUEST_NULL;
            MPI_Irecv(donor_local_ids.data() + offset, count, mpi_local_type,
                      neighbor, 4318, comm, &req);
            reqs.push_back(req);
        }
    }

    for (int k = 0; k < n_nb; ++k) {
        const std::size_t uk = static_cast<std::size_t>(k);
        const int neighbor = mp.nb_ranks[uk];
        const cfd::LocalIndex offset = mp.send_offsets[uk];
        const int count = static_cast<int>(mp.send_offsets[uk + 1] - offset);
        if (count > 0) {
            MPI_Request req = MPI_REQUEST_NULL;
            MPI_Isend(const_cast<cfd::LocalIndex*>(mp.send_owned_local.data() + offset),
                      count, mpi_local_type, neighbor, 4318, comm, &req);
            reqs.push_back(req);
        }
    }

    if (!reqs.empty()) {
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<cfd::GlobalIndex> ghost_gids;
    ghost_gids.reserve(mp.recv_ghost_local.size());

    for (int k = 0; k < n_nb; ++k) {
        const std::size_t uk = static_cast<std::size_t>(k);
        const int donor_rank = mp.nb_ranks[uk];
        const cfd::GlobalIndex donor_offset = rank_offsets[static_cast<std::size_t>(donor_rank)];
        const cfd::LocalIndex begin_idx = mp.recv_offsets[uk];
        const cfd::LocalIndex end_idx = mp.recv_offsets[uk + 1];

        for (cfd::LocalIndex i = begin_idx; i < end_idx; ++i) {
            const cfd::LocalIndex donor_local = donor_local_ids[static_cast<std::size_t>(i)];
            ghost_gids.push_back(donor_offset + static_cast<cfd::GlobalIndex>(donor_local));
        }
    }

    std::sort(ghost_gids.begin(), ghost_gids.end());
    ghost_gids.erase(std::unique(ghost_gids.begin(), ghost_gids.end()), ghost_gids.end());
    return ghost_gids;
}

void factor_diagonal_blocks(const cfd::linalg::BsrMatrix& mat, int bs,
                            std::vector<double>& diag_lu,
                            std::vector<cfd::LocalIndex>& diag_piv) {
    const cfd::LocalIndex n_own = mat.localRows();
    const std::size_t bs_sz = static_cast<std::size_t>(bs);
    const std::size_t bs2 = bs_sz * bs_sz;

    diag_lu.resize(static_cast<std::size_t>(n_own) * bs2);
    diag_piv.resize(static_cast<std::size_t>(n_own) * bs_sz);

    const auto& diag = mat.diagIndex();
    const double* val = mat.valuesData();

    for (cfd::LocalIndex i = 0; i < n_own; ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        const std::size_t d = static_cast<std::size_t>(diag[ui]);
        const double* diag_src = val + d * bs2;
        double* diag_dst = diag_lu.data() + ui * bs2;

        std::copy(diag_src, diag_src + bs2, diag_dst);
        cfd::linalg::detail::lu_factor_block(diag_dst, bs, diag_piv.data() + ui * bs_sz);
    }
}

void execute_sgs_iteration(const cfd::linalg::BsrMatrix& mat, int bs,
                           const std::vector<double>& diag_lu,
                           const std::vector<cfd::LocalIndex>& diag_piv,
                           const cfd::linalg::Vector& rhs,
                           cfd::linalg::Vector& x) {
    const cfd::LocalIndex n_own = mat.localRows();

    // 1. Forward sweep
    x.updateGhosts();
    cfd::linalg::detail::bsr_sgs_sweep_dispatch(
        n_own, true, bs,
        mat.rowPtr().data(), mat.cols().data(), mat.valuesData(),
        mat.diagIndex().data(), diag_lu.data(), diag_piv.data(),
        rhs.data(), x.data()
    );

    // 2. Backward sweep
    x.updateGhosts();
    cfd::linalg::detail::bsr_sgs_sweep_dispatch(
        n_own, false, bs,
        mat.rowPtr().data(), mat.cols().data(), mat.valuesData(),
        mat.diagIndex().data(), diag_lu.data(), diag_piv.data(),
        rhs.data(), x.data()
    );
}

} // namespace

void run_vector_tests(const cfd::mesh::MeshPart& mp) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::printf("\n======================================================================\n");
        std::printf("           DISTRIBUTED VECTOR ANALYTICAL TEST SUITE                  \n");
        std::printf("======================================================================\n");
    }

    // 1. Build VectorLayout based on cell partition
    cfd::linalg::VectorLayout layout(comm, mp.n_cells_g, mp.n_own);

    // Collect unique global IDs of ghost cells
    std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mp, comm);
    layout.setGhosts(ghost_gids);

    const double N_g = static_cast<double>(mp.n_cells_g);
    const cfd::GlobalIndex row_begin = layout.localBegin();
    const cfd::LocalIndex n_own = layout.localSize();

    // -------------------------------------------------------------------------
    // TEST 1: All-Ones Dot Product (Sum of all cells)
    // x = [1, 1, ..., 1], y = [1, 1, ..., 1] -> x . y = N_global
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector x(layout, 1);
        cfd::linalg::Vector y(layout, 1);

        std::fill(x.owned().begin(), x.owned().end(), 1.0);
        std::fill(y.owned().begin(), y.owned().end(), 1.0);

        const double computed = x.dot(y);
        const double analytical = N_g;
        report_test("Dot Product (All-Ones: x . y = N)", computed, analytical, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 2: Euclidean Norm of Constant Vector
    // x = [3, 3, ..., 3] -> ||x||_2 = sqrt(9 * N_global) = 3 * sqrt(N_global)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector x(layout, 1);
        std::fill(x.owned().begin(), x.owned().end(), 3.0);

        const double computed = x.norm2();
        const double analytical = 3.0 * std::sqrt(N_g);
        report_test("Norm2 of Constant Vector (||c||_2 = c * sqrt(N))", computed, analytical, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 3: Arithmetic Progression Dot Product
    // x = [1, 1, ...], y = [1, 2, ..., N] -> x . y = N * (N + 1) / 2
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector x(layout, 1);
        cfd::linalg::Vector y(layout, 1);

        auto x_data = x.owned();
        auto y_data = y.owned();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            x_data[idx] = 1.0;
            y_data[idx] = static_cast<double>(row_begin) + static_cast<double>(i) + 1.0;
        }

        const double computed = x.dot(y);
        const double analytical = 0.5 * N_g * (N_g + 1.0);
        report_test("Arithmetic Progression Dot (x . y = N(N+1)/2)", computed, analytical, 1e-7 * analytical, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 4: Strictly Orthogonal Vectors
    // u = [k - (N-1)/2], v = [1, 1, ...] -> u . v = 0.0
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector u(layout, 1);
        cfd::linalg::Vector v(layout, 1);

        auto u_data = u.owned();
        auto v_data = v.owned();
        const double mean_idx = 0.5 * (N_g - 1.0);

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            u_data[idx] = (static_cast<double>(row_begin) + static_cast<double>(i)) - mean_idx;
            v_data[idx] = 1.0;
        }

        const double computed = u.dot(v);
        const double analytical = 0.0;
        report_test("Orthogonal Vectors Inner Product (u . v = 0)", computed, analytical, 1e-9 * N_g, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 5: AXPY Linearity and Residual Elimination
    // y = 6.0, x = 2.0; y <- y + (-3.0) * x => y = 0, ||y||_2 = 0.0
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector x(layout, 1);
        cfd::linalg::Vector y(layout, 1);

        std::fill(x.owned().begin(), x.owned().end(), 2.0);
        std::fill(y.owned().begin(), y.owned().end(), 6.0);

        y.axpy(-3.0, x);

        const double computed = y.norm2();
        const double analytical = 0.0;
        report_test("AXPY Cancellation (y + alpha*x = 0, ||y||_2 = 0)", computed, analytical, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 6: Halo Exchange (updateGhosts)
    // Populate owned slots with global_row + 42.0.
    // Verify ghost slots match the expected values after communication.
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector x(layout, 1);
        auto x_owned = x.owned();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            x_owned[idx] = static_cast<double>(row_begin) + static_cast<double>(i) + 42.0;
        }

        std::fill(x.ghosts().begin(), x.ghosts().end(), 0.0);

        x.updateGhosts();

        const auto& ghost_ids = layout.ghostGlobalIds();
        auto x_ghosts = x.ghosts();
        double local_max_diff = 0.0;

        for (std::size_t g = 0; g < ghost_ids.size(); ++g) {
            const double expected = static_cast<double>(ghost_ids[g]) + 42.0;
            local_max_diff = std::max(local_max_diff, std::abs(x_ghosts[g] - expected));
        }

        double global_max_diff = 0.0;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, comm);

        report_test("Ghost Exchange Correctness (max |ghost[i] - expected|)", global_max_diff, 0.0, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 7: Block-Vector Norm and Ghost Exchange (block_size = 3)
    // v = [1.0, 2.0, 3.0] per row -> ||v||_2 = sqrt(N_g * (1^2 + 2^2 + 3^2)) = sqrt(14 * N_g)
    // -------------------------------------------------------------------------
    {
        const int bs = 3;
        const std::size_t bs_sz = static_cast<std::size_t>(bs);
        cfd::linalg::Vector v(layout, bs);
        auto v_owned = v.owned();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * bs_sz;
            v_owned[base + 0] = 1.0;
            v_owned[base + 1] = 2.0;
            v_owned[base + 2] = 3.0;
        }

        const double computed_norm = v.norm2();
        const double analytical_norm = std::sqrt(14.0 * N_g);
        report_test("Block Vector Norm2 (bs=3, ||v||_2 = sqrt(14*N))", computed_norm, analytical_norm, 1e-12, comm);

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * bs_sz;
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            v_owned[base + 0] = gid * 10.0 + 1.0;
            v_owned[base + 1] = gid * 10.0 + 2.0;
            v_owned[base + 2] = gid * 10.0 + 3.0;
        }

        v.updateGhosts();

        const auto& ghost_ids = layout.ghostGlobalIds();
        auto v_ghosts = v.ghosts();
        double local_max_diff = 0.0;

        for (std::size_t g = 0; g < ghost_ids.size(); ++g) {
            const double gid = static_cast<double>(ghost_ids[g]);
            const std::size_t base = g * bs_sz;
            local_max_diff = std::max(local_max_diff, std::abs(v_ghosts[base + 0] - (gid * 10.0 + 1.0)));
            local_max_diff = std::max(local_max_diff, std::abs(v_ghosts[base + 1] - (gid * 10.0 + 2.0)));
            local_max_diff = std::max(local_max_diff, std::abs(v_ghosts[base + 2] - (gid * 10.0 + 3.0)));
        }

        double global_max_diff = 0.0;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, comm);

        report_test("Block Ghost Exchange Correctness (bs=3, max err)", global_max_diff, 0.0, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 8: Batched Inner Products (Fused Allreduce)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::Vector a(layout, 1);
        cfd::linalg::Vector b(layout, 1);
        cfd::linalg::Vector c(layout, 1);

        std::fill(a.owned().begin(), a.owned().end(), 2.0);
        std::fill(b.owned().begin(), b.owned().end(), 3.0);
        std::fill(c.owned().begin(), c.owned().end(), 4.0);

        std::vector<std::pair<const cfd::linalg::Vector*, const cfd::linalg::Vector*>> pairs = {
            {&a, &b},
            {&a, &c},
            {&b, &c}
        };
        std::vector<double> results(3, 0.0);

        cfd::linalg::Vector::batchedDots(pairs, results);

        const double diff = std::abs(results[0] - 6.0 * N_g) +
                            std::abs(results[1] - 8.0 * N_g) +
                            std::abs(results[2] - 12.0 * N_g);

        report_test("Batched Dots Fusion (batchedDots vs expected)", diff, 0.0, 1e-11 * N_g, comm);
    }

    if (rank == 0) {
        std::printf("======================================================================\n\n");
    }
}

void run_bsr_matrix_tests(const cfd::mesh::MeshPart& mp) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::printf("\n======================================================================\n");
        std::printf("             BSR MATRIX ANALYTICAL ASSEMBLY & MATVEC SUITE            \n");
        std::printf("======================================================================\n");
    }

    cfd::mesh::MeshAuxConnectivity aux = 
        cfd::mesh::build_aux_connectivity(mp, cfd::mesh::AuxConnType::CellCellsByFace);
    const std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mp, comm);

    const double N_g = static_cast<double>(mp.n_cells_g);
    const cfd::LocalIndex n_own = mp.n_own;

    // TEST 1: Dual Graph Matrix Assembly & Diagonal Placement Invariants
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        int local_invariants_pass = 1;
        if (!mat.assembled()) {
            local_invariants_pass = 0;
        }

        const auto& rptr = mat.rowPtr();
        const auto& cols = mat.cols();
        const auto& diag = mat.diagIndex();

        if (rptr.size() != static_cast<std::size_t>(n_own + 1) || diag.size() != static_cast<std::size_t>(n_own)) {
            local_invariants_pass = 0;
        }

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex d = diag[ui];
            if (d < rptr[ui] || d >= rptr[ui + 1] || cols[static_cast<std::size_t>(d)] != i) {
                local_invariants_pass = 0;
                break;
            }
        }

        int global_invariants_pass = 0;
        MPI_Allreduce(&local_invariants_pass, &global_invariants_pass, 1, MPI_INT, MPI_MIN, comm);
        report_test("BSR Structural Assembly & Diagonal Integrity",
                    static_cast<double>(global_invariants_pass), 1.0, 0.0, comm);
    }

    // TEST 2: Identity MatVec (bs = 1, A = I, y = A * x)
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const std::size_t total_nnz = mat.values().size();
        std::fill(val, val + total_nnz, 0.0);

        const auto& diag = mat.diagIndex();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t d = static_cast<std::size_t>(diag[static_cast<std::size_t>(i)]);
            val[d] = 1.0;
        }

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();
        auto x_owned = x.owned();

        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            x_owned[ui] = static_cast<double>(row_begin) + static_cast<double>(i) + 1.0;
        }

        x.updateGhosts();
        mat.apply(x, y, 1.0, 0.0);

        y.axpy(-1.0, x);
        const double error_norm = y.norm2();
        report_test("Identity MatVec SpMV (bs=1, ||A*x - x||_2)", error_norm, 0.0, 1e-12, comm);
    }

    // TEST 3: Graph Laplacian Nullspace (bs = 1, L * 1 = 0)
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                val[uk] = (k == d_k) ? degree : -1.0;
            }
        }

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();
        std::fill(x.owned().begin(), x.owned().end(), 1.0);

        x.updateGhosts();
        mat.apply(x, y, 1.0, 0.0);

        const double error_norm = y.norm2();
        report_test("Graph Laplacian Nullspace (bs=1, ||L*1||_2 = 0)", error_norm, 0.0, 1e-12, comm);
    }

    // TEST 4: Full Non-Zero Accumulation & Degree Row-Sum (bs = 1, all entries = 1)
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 1.0);

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();
        std::fill(x.owned().begin(), x.owned().end(), 1.0);

        x.updateGhosts();
        mat.apply(x, y, 1.0, 0.0);

        const auto& rptr = mat.rowPtr();
        auto y_owned = y.owned();
        double local_diff_sq = 0.0;

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const double expected_deg = static_cast<double>(rptr[ui + 1] - rptr[ui]);
            const double diff = y_owned[ui] - expected_deg;
            local_diff_sq += diff * diff;
        }

        double global_diff_sq = 0.0;
        MPI_Allreduce(&local_diff_sq, &global_diff_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
        const double error_norm = std::sqrt(global_diff_sq);

        report_test("All-Ones Row Accumulation (bs=1, ||A*1 - deg||_2)", error_norm, 0.0, 1e-12, comm);
    }

    // TEST 5: Block-Matrix Identity (bs = 4, A = I_4 per cell)
    {
        const int bs = 4;
        const std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& diag = mat.diagIndex();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t d = static_cast<std::size_t>(diag[static_cast<std::size_t>(i)]);
            const std::size_t base = d * blk_stride;
            for (int b = 0; b < bs; ++b) {
                const std::size_t ub = static_cast<std::size_t>(b);
                val[base + ub * static_cast<std::size_t>(bs) + ub] = 1.0;
            }
        }

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();
        auto x_owned = x.owned();

        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            for (int b = 0; b < bs; ++b) {
                x_owned[base + static_cast<std::size_t>(b)] = gid * 10.0 + static_cast<double>(b + 1);
            }
        }

        x.updateGhosts();
        mat.apply(x, y, 1.0, 0.0);

        y.axpy(-1.0, x);
        const double error_norm = y.norm2();
        report_test("Block Identity MatVec SpMV (bs=4, ||A*x - x||_2)", error_norm, 0.0, 1e-12, comm);
    }

    // TEST 6: Coupled Vector Graph Laplacian Nullspace (bs = 3)
    {
        const int bs = 3;
        const std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t base = static_cast<std::size_t>(k) * blk_stride;
                const double factor = (k == d_k) ? degree : -1.0;
                for (int b = 0; b < bs; ++b) {
                    const std::size_t ub = static_cast<std::size_t>(b);
                    val[base + ub * static_cast<std::size_t>(bs) + ub] = factor;
                }
            }
        }

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();
        auto x_owned = x.owned();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            x_owned[base + 0] = 1.0;
            x_owned[base + 1] = 2.0;
            x_owned[base + 2] = 3.0;
        }

        x.updateGhosts();
        mat.apply(x, y, 1.0, 0.0);

        const double error_norm = y.norm2();
        report_test("Coupled Laplacian Nullspace (bs=3, ||L*x||_2 = 0)", error_norm, 0.0, 1e-12, comm);
    }

    // TEST 7: Operator Linear Combination (y <- alpha * A * x + beta * y)
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& diag = mat.diagIndex();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t d = static_cast<std::size_t>(diag[static_cast<std::size_t>(i)]);
            val[d] = 1.0;
        }

        cfd::linalg::Vector x = mat.makeVector();
        cfd::linalg::Vector y = mat.makeVector();

        std::fill(x.owned().begin(), x.owned().end(), 2.0);
        std::fill(y.owned().begin(), y.owned().end(), 5.0);

        x.updateGhosts();
        mat.apply(x, y, 3.0, -2.0);

        const double computed_norm = y.norm2();
        const double analytical_norm = 4.0 * std::sqrt(N_g);
        report_test("Linear Combination MatVec (y <- alpha*A*x + beta*y)",
                    computed_norm, analytical_norm, 1e-12, comm);
    }

    if (rank == 0) {
        std::printf("======================================================================\n\n");
    }
}

void run_bsr_sgs_tests(const cfd::mesh::MeshPart& mp) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::printf("\n======================================================================\n");
        std::printf("             BSR SGS (SYMMETRIC GAUSS-SEIDEL) TEST SUITE              \n");
        std::printf("======================================================================\n");
    }

    cfd::mesh::MeshAuxConnectivity aux =
        cfd::mesh::build_aux_connectivity(mp, cfd::mesh::AuxConnType::CellCellsByFace);
    const std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mp, comm);

    const cfd::LocalIndex n_own = mp.n_own;

    // -------------------------------------------------------------------------
    // TEST 1: Block LU Factorization & Substitution Invariant (bs = 3)
    // -------------------------------------------------------------------------
    {
        constexpr int bs = 3;
        const double a_orig[9] = {
            4.0, 1.0, 0.5,
            1.0, 5.0, 1.0,
            0.5, 1.0, 3.0
        };
        double a_lu[9];
        std::copy(a_orig, a_orig + 9, a_lu);

        cfd::LocalIndex piv[3] = {0, 0, 0};
        const bool factor_ok = cfd::linalg::detail::lu_factor_block(a_lu, bs, piv);

        const double x_exact[3] = {1.0, 2.0, 1.5};
        double rhs[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < bs; ++j) {
                rhs[i] += a_orig[i * bs + j] * x_exact[j];
            }
        }

        cfd::linalg::detail::lu_solve_block<bs>(a_lu, piv, rhs);

        const double err = std::abs(rhs[0] - x_exact[0]) + 
                           std::abs(rhs[1] - x_exact[1]) + 
                           std::abs(rhs[2] - x_exact[2]);
        const double computed = factor_ok ? err : 1.0;
        report_test("Block LU Factor and Solve Correctness (bs=3)", computed, 0.0, 1e-13, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 2: Scalar SGS Solver Convergence (bs = 1, A = L + 2*I, x* = 1.0)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                val[uk] = (k == d_k) ? (degree + 2.0) : -1.0;
            }
        }

        std::vector<double> diag_lu;
        std::vector<cfd::LocalIndex> diag_piv;
        factor_diagonal_blocks(mat, 1, diag_lu, diag_piv);

        cfd::linalg::Vector x_exact = mat.makeVector();
        std::fill(x_exact.owned().begin(), x_exact.owned().end(), 1.0);
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        const double r0_norm = b.norm2();

        constexpr int n_iter = 25;
        for (int iter = 0; iter < n_iter; ++iter) {
            execute_sgs_iteration(mat, 1, diag_lu, diag_piv, b, x);
        }

        x.updateGhosts();
        cfd::linalg::Vector Ax = mat.makeVector();
        mat.apply(x, Ax, 1.0, 0.0);
        cfd::linalg::Vector res = mat.makeVector();
        res.copyFrom(b);
        res.axpy(-1.0, Ax);

        const double r_norm = res.norm2();
        const double reduction = r_norm / r0_norm;

        report_test("Scalar SGS Residual Reduction (bs=1, ||r||/||r0|| <= 1e-4)", reduction, 0.0, 1e-4, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_err = err.norm2() / x_exact.norm2();
        report_test("Scalar SGS Solution Error (bs=1, ||x - x*||/||x*|| <= 1e-4)", rel_err, 0.0, 1e-4, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 3: Block SGS Coupled System Convergence (bs = 3)
    // -------------------------------------------------------------------------
    {
        constexpr int bs = 3;
        constexpr std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t base = static_cast<std::size_t>(k) * blk_stride;
                if (k == d_k) {
                    val[base + 0] = degree + 3.0; val[base + 1] = 0.4;          val[base + 2] = 0.2;
                    val[base + 3] = 0.4;          val[base + 4] = degree + 3.0; val[base + 5] = 0.3;
                    val[base + 6] = 0.2;          val[base + 7] = 0.3;          val[base + 8] = degree + 3.0;
                } else {
                    for (int b = 0; b < bs; ++b) {
                        const std::size_t ub = static_cast<std::size_t>(b);
                        val[base + ub * static_cast<std::size_t>(bs) + ub] = -1.0;
                    }
                }
            }
        }

        std::vector<double> diag_lu;
        std::vector<cfd::LocalIndex> diag_piv;
        factor_diagonal_blocks(mat, bs, diag_lu, diag_piv);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            x_exact_owned[base + 0] = 1.0;
            x_exact_owned[base + 1] = 2.0;
            x_exact_owned[base + 2] = 3.0;
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        const double r0_norm = b.norm2();

        constexpr int n_iter = 25;
        for (int iter = 0; iter < n_iter; ++iter) {
            execute_sgs_iteration(mat, bs, diag_lu, diag_piv, b, x);
        }

        x.updateGhosts();
        cfd::linalg::Vector Ax = mat.makeVector();
        mat.apply(x, Ax, 1.0, 0.0);
        cfd::linalg::Vector res = mat.makeVector();
        res.copyFrom(b);
        res.axpy(-1.0, Ax);

        const double r_norm = res.norm2();
        const double reduction = r_norm / r0_norm;

        report_test("Block Coupled SGS Residual Reduction (bs=3, ||r||/||r0|| <= 1e-4)", reduction, 0.0, 1e-4, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 4: Block SGS Monotonic Residual Drop (bs = 4)
    // -------------------------------------------------------------------------
    {
        constexpr int bs = 4;
        constexpr std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t base = static_cast<std::size_t>(k) * blk_stride;
                const double factor = (k == d_k) ? (degree + 3.0) : -1.0;
                for (int b = 0; b < bs; ++b) {
                    const std::size_t ub = static_cast<std::size_t>(b);
                    val[base + ub * static_cast<std::size_t>(bs) + ub] = factor;
                }
            }
        }

        std::vector<double> diag_lu;
        std::vector<cfd::LocalIndex> diag_piv;
        factor_diagonal_blocks(mat, bs, diag_lu, diag_piv);

        cfd::linalg::Vector b = mat.makeVector();
        std::fill(b.owned().begin(), b.owned().end(), 1.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        double prev_norm = b.norm2();
        int monotonic_drop_local = 1;

        constexpr int n_iter = 20;
        for (int iter = 0; iter < n_iter; ++iter) {
            execute_sgs_iteration(mat, bs, diag_lu, diag_piv, b, x);

            x.updateGhosts();
            cfd::linalg::Vector Ax = mat.makeVector();
            mat.apply(x, Ax, 1.0, 0.0);
            cfd::linalg::Vector res = mat.makeVector();
            res.copyFrom(b);
            res.axpy(-1.0, Ax);

            const double cur_norm = res.norm2();
            if (cur_norm > prev_norm + 1e-12) {
                monotonic_drop_local = 0;
            }
            prev_norm = cur_norm;
        }

        int monotonic_drop_global = 0;
        MPI_Allreduce(&monotonic_drop_local, &monotonic_drop_global, 1, MPI_INT, MPI_MIN, comm);
        report_test("Block SGS Monotonic Residual Decrease (bs=4)",
                    static_cast<double>(monotonic_drop_global), 1.0, 0.0, comm);
    }

    if (rank == 0) {
        std::printf("======================================================================\n\n");
    }
}

void run_bicgstab_tests(const cfd::mesh::MeshPart& mp) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::printf("\n======================================================================\n");
        std::printf("             DISTRIBUTED BiCGSTAB + IDENTITY PRECONDITIONER SUITE     \n");
        std::printf("======================================================================\n");
    }

    cfd::mesh::MeshAuxConnectivity aux = 
        cfd::mesh::build_aux_connectivity(mp, cfd::mesh::AuxConnType::CellCellsByFace);
    const std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mp, comm);

    const cfd::LocalIndex n_own = mp.n_own;

    // -------------------------------------------------------------------------
    // TEST 1: Identity Operator Exact Step Convergence (bs = 1, A = I)
    // BiCGSTAB with M = I must converge in exactly 1 iteration.
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& diag = mat.diagIndex();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t d = static_cast<std::size_t>(diag[static_cast<std::size_t>(i)]);
            val[d] = 1.0;
        }

        cfd::linalg::IdentityPreconditioner prec;
        prec.setup(mat);

        cfd::linalg::Vector b = mat.makeVector();
        auto b_owned = b.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            b_owned[static_cast<std::size_t>(i)] = static_cast<double>(row_begin) + static_cast<double>(i) + 1.0;
        }

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, prec, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        const double conv_metric = converged ? 1.0 : 0.0;
        report_test("Identity BiCGSTAB Convergence Flag", conv_metric, 1.0, 0.0, comm);

        const double iters = static_cast<double>(result.iterations);
        report_test("Identity BiCGSTAB 1-Iteration Exit", iters, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, b);
        const double sol_err = err.norm2();
        report_test("Identity BiCGSTAB Solution Accuracy (||x - b||_2)", sol_err, 0.0, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 2: Symmetric Positive Definite Shifted Laplacian (bs = 1, A = L + 3*I)
    // Uses non-constant x* to avoid degenerate single-eigenmode breakdown.
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                val[uk] = (k == d_k) ? (degree + 3.0) : -1.0;
            }
        }

        cfd::linalg::IdentityPreconditioner prec;
        prec.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            x_exact_owned[static_cast<std::size_t>(i)] = 1.0 + std::cos(gid * 0.2);
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, prec, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("SPD Shifted Laplacian Convergence Flag", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("SPD Shifted Laplacian Relative Error (||x - x*||/||x*||)", rel_sol_err, 0.0, 1e-7, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 3: Non-Symmetric Convective System (bs = 1, A_ij != A_ji)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& cols = mat.cols();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];

            double row_abs_off_sum = 0.0;
            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                if (k == d_k) continue;
                const std::size_t uk = static_cast<std::size_t>(k);
                const cfd::LocalIndex j = cols[uk];
                const double edge_weight = (j > i) ? -1.6 : -0.6;
                val[uk] = edge_weight;
                row_abs_off_sum += std::abs(edge_weight);
            }
            val[static_cast<std::size_t>(d_k)] = row_abs_off_sum + 2.5;
        }

        cfd::linalg::IdentityPreconditioner prec;
        prec.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            x_exact_owned[static_cast<std::size_t>(i)] = 1.0 + std::sin(gid * 0.1);
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, prec, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("Non-Symmetric Matrix Convergence Flag", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("Non-Symmetric Relative Error (||x - x*||/||x*||)", rel_sol_err, 0.0, 1e-7, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 4: Block Coupled Non-Symmetric System (bs = 3)
    // -------------------------------------------------------------------------
    {
        constexpr int bs = 3;
        constexpr std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& rptr = mat.rowPtr();
        const auto& cols = mat.cols();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t base = static_cast<std::size_t>(k) * blk_stride;
                if (k == d_k) {
                    const double degree = static_cast<double>(end_k - start_k - 1);
                    val[base + 0] = degree + 4.0; val[base + 1] =  0.6;         val[base + 2] = -0.4;
                    val[base + 3] = -0.5;         val[base + 4] = degree + 4.0; val[base + 5] =  0.5;
                    val[base + 6] =  0.3;         val[base + 7] = -0.3;         val[base + 8] = degree + 4.0;
                } else {
                    const cfd::LocalIndex j = cols[static_cast<std::size_t>(k)];
                    const double factor = (j > i) ? -1.3 : -0.7;
                    for (int b = 0; b < bs; ++b) {
                        const std::size_t ub = static_cast<std::size_t>(b);
                        val[base + ub * static_cast<std::size_t>(bs) + ub] = factor;
                    }
                }
            }
        }

        cfd::linalg::IdentityPreconditioner prec;
        prec.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            x_exact_owned[base + 0] = 1.0;
            x_exact_owned[base + 1] = 2.0;
            x_exact_owned[base + 2] = 3.0;
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, prec, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("Block Coupled Non-Symmetric Convergence Flag (bs=3)", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("Block Coupled Non-Symmetric Error (bs=3, ||x - x*||/||x*||)", rel_sol_err, 0.0, 1e-7, comm);
    }

    if (rank == 0) {
        std::printf("======================================================================\n\n");
    }
}

void run_bicgstab_sgs_tests(const cfd::mesh::MeshPart& mp) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::printf("\n======================================================================\n");
        std::printf("             DISTRIBUTED BiCGSTAB + SGS PRECONDITIONER SUITE          \n");
        std::printf("======================================================================\n");
    }

    cfd::mesh::MeshAuxConnectivity aux = 
        cfd::mesh::build_aux_connectivity(mp, cfd::mesh::AuxConnType::CellCellsByFace);
    const std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mp, comm);

    const cfd::LocalIndex n_own = mp.n_own;

    // -------------------------------------------------------------------------
    // TEST 1: SGS Preconditioner Contract Verification (Halo Exchange on Exit)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& diag = mat.diagIndex();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t d = static_cast<std::size_t>(diag[static_cast<std::size_t>(i)]);
            val[d] = 2.0;
        }

        cfd::linalg::SgsPreconditioner sgs;
        sgs.setup(mat);

        cfd::linalg::Vector r = mat.makeVector();
        cfd::linalg::Vector z = mat.makeVector();

        auto r_owned = r.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            r_owned[static_cast<std::size_t>(i)] = static_cast<double>(row_begin) + static_cast<double>(i) + 1.0;
        }

        // Apply preconditioner z = M^{-1} r
        sgs.apply(r, z);

        // Verify that ghosts in z were updated during apply (contract check)
        const auto& ghost_ids = mat.layout().ghostGlobalIds();
        auto z_ghosts = z.ghosts();
        double local_max_diff = 0.0;

        for (std::size_t g = 0; g < ghost_ids.size(); ++g) {
            const double expected = (static_cast<double>(ghost_ids[g]) + 1.0) / 2.0;
            local_max_diff = std::max(local_max_diff, std::abs(z_ghosts[g] - expected));
        }

        double global_max_diff = 0.0;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, comm);
        report_test("SGS Apply Contract (Ghost Exchange on Exit)", global_max_diff, 0.0, 1e-12, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 2: Symmetric Shifted Laplacian (bs = 1, A = L + 3*I)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                val[uk] = (k == d_k) ? (degree + 3.0) : -1.0;
            }
        }

        cfd::linalg::SgsPreconditioner sgs;
        sgs.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            x_exact_owned[static_cast<std::size_t>(i)] = 1.0 + std::cos(gid * 0.2);
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, sgs, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("SPD Shifted Laplacian SGS Convergence Flag", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("SPD Shifted Laplacian SGS Error (||x - x*||/||x*||)", rel_sol_err, 0.0, 1e-7, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 3: Non-Symmetric Convective Matrix (bs = 1)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& cols = mat.cols();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];

            double row_abs_off_sum = 0.0;
            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                if (k == d_k) continue;
                const std::size_t uk = static_cast<std::size_t>(k);
                const cfd::LocalIndex j = cols[uk];
                const double edge_weight = (j > i) ? -1.6 : -0.6;
                val[uk] = edge_weight;
                row_abs_off_sum += std::abs(edge_weight);
            }
            val[static_cast<std::size_t>(d_k)] = row_abs_off_sum + 2.5;
        }

        cfd::linalg::SgsPreconditioner sgs;
        sgs.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            x_exact_owned[static_cast<std::size_t>(i)] = 1.0 + std::sin(gid * 0.1);
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, sgs, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("Non-Symmetric Matrix SGS Convergence Flag", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("Non-Symmetric SGS Error (||x - x*||/||x*||)", rel_sol_err, 0.0, 1e-7, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 4: Block Coupled Non-Symmetric System (bs = 3)
    // -------------------------------------------------------------------------
    {
        constexpr int bs = 3;
        constexpr std::size_t blk_stride = static_cast<std::size_t>(bs * bs);
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, bs);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        std::fill(val, val + mat.values().size(), 0.0);

        const auto& rptr = mat.rowPtr();
        const auto& cols = mat.cols();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t base = static_cast<std::size_t>(k) * blk_stride;
                if (k == d_k) {
                    const double degree = static_cast<double>(end_k - start_k - 1);
                    val[base + 0] = degree + 4.0; val[base + 1] =  0.6;         val[base + 2] = -0.4;
                    val[base + 3] = -0.5;         val[base + 4] = degree + 4.0; val[base + 5] =  0.5;
                    val[base + 6] =  0.3;         val[base + 7] = -0.3;         val[base + 8] = degree + 4.0;
                } else {
                    const cfd::LocalIndex j = cols[static_cast<std::size_t>(k)];
                    const double factor = (j > i) ? -1.3 : -0.7;
                    for (int b = 0; b < bs; ++b) {
                        const std::size_t ub = static_cast<std::size_t>(b);
                        val[base + ub * static_cast<std::size_t>(bs) + ub] = factor;
                    }
                }
            }
        }

        cfd::linalg::SgsPreconditioner sgs;
        sgs.setup(mat);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            x_exact_owned[base + 0] = 1.0;
            x_exact_owned[base + 1] = 2.0;
            x_exact_owned[base + 2] = 3.0;
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, sgs, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("Block Coupled Non-Symmetric SGS Convergence (bs=3)", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("Block Coupled Non-Symmetric SGS Error (bs=3)", rel_sol_err, 0.0, 1e-7, comm);
    }

    // -------------------------------------------------------------------------
    // TEST 5: SGS Multi-Sweep Verification (sweeps = 2)
    // -------------------------------------------------------------------------
    {
        cfd::linalg::BsrMatrix mat(comm, mp.n_cells_g, mp.n_own, 1);
        mat.assemble(ghost_gids, aux.cell_cells_face_offsets, aux.cell_cells_face);

        double* val = mat.valuesData();
        const auto& rptr = mat.rowPtr();
        const auto& diag = mat.diagIndex();

        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const cfd::LocalIndex start_k = rptr[ui];
            const cfd::LocalIndex end_k = rptr[ui + 1];
            const cfd::LocalIndex d_k = diag[ui];
            const double degree = static_cast<double>(end_k - start_k - 1);

            for (cfd::LocalIndex k = start_k; k < end_k; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                val[uk] = (k == d_k) ? (degree + 3.0) : -1.0;
            }
        }

        cfd::linalg::SgsPreconditioner sgs;
        sgs.setup(mat);
        sgs.setSweeps(2);

        cfd::linalg::Vector x_exact = mat.makeVector();
        auto x_exact_owned = x_exact.owned();
        const cfd::GlobalIndex row_begin = mat.layout().localBegin();
        for (cfd::LocalIndex i = 0; i < n_own; ++i) {
            const double gid = static_cast<double>(row_begin) + static_cast<double>(i);
            x_exact_owned[static_cast<std::size_t>(i)] = 1.0 + std::cos(gid * 0.2);
        }
        x_exact.updateGhosts();

        cfd::linalg::Vector b = mat.makeVector();
        mat.apply(x_exact, b, 1.0, 0.0);

        cfd::linalg::Vector x = mat.makeVector();
        x.setZero();

        cfd::linalg::BiCGSTAB solver;
        cfd::linalg::IterationResult result = solver.solve(mat, sgs, x, b);

        const bool converged = (result.status == cfd::linalg::SolverStatus::Converged);
        report_test("SGS Multi-Sweep (sweeps=2) Convergence Flag", converged ? 1.0 : 0.0, 1.0, 0.0, comm);

        cfd::linalg::Vector err = mat.makeVector();
        err.copyFrom(x);
        err.axpy(-1.0, x_exact);
        const double rel_sol_err = err.norm2() / x_exact.norm2();
        report_test("SGS Multi-Sweep (sweeps=2) Solution Error", rel_sol_err, 0.0, 1e-7, comm);
    }

    if (rank == 0) {
        std::printf("======================================================================\n\n");
    }
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI did not provide requested thread level\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string mesh_file;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-v" || a == "--verbose") {
            verbose = 1;
        } else if (a == "-vv") {
            verbose = 2;
        } else if (mesh_file.empty()) {
            mesh_file = a;
        } else {
            if (rank == 0) {
                std::fprintf(stderr, "usage: mpirun -np N %s <mesh.h5> [-v|-vv]\n", argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (mesh_file.empty()) {
        if (rank == 0) {
            std::fprintf(stderr, "usage: mpirun -np N %s <mesh.h5> [-v|-vv]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cfd::mpi::log_init(verbose);
    cfd::mpi::log_info("test_runner: mesh=%s ranks=%d", mesh_file.c_str(), nprocs);

    const double t0 = MPI_Wtime();

    cfd::mesh::MeshPart mp;
    cfd::io::solver_mesh::import_mesh_hdf5(mp, mesh_file, MPI_COMM_WORLD);
    cfd::mesh::validate_and_log_meshpart(mp);

    run_vector_tests(mp);
    run_bsr_matrix_tests(mp);
    run_bsr_sgs_tests(mp);
    run_bicgstab_tests(mp);
    run_bicgstab_sgs_tests(mp);

    if (rank == 0) {
        std::fprintf(stderr, "Total test execution time = %.5f sec\n", MPI_Wtime() - t0);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
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
#include "cfd/linalg/vector.hpp"
#include "cfd/linalg/vector_layout.hpp"

#include "cfd/io/solver_mesh/hdf5_reader.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/validate.hpp"
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
    // -------------------------------------------------------------------------
    // 1. Gather n_own from all ranks to compute global ownership offsets
    // -------------------------------------------------------------------------
    const MPI_Datatype mpi_local_type = (sizeof(cfd::LocalIndex) == 8) ? MPI_INT64_T : MPI_INT32_T;

    std::vector<cfd::LocalIndex> all_n_own(static_cast<std::size_t>(mp.nprocs), 0);
    MPI_Allgather(&mp.n_own, 1, mpi_local_type,
                  all_n_own.data(), 1, mpi_local_type, comm);

    std::vector<cfd::GlobalIndex> rank_offsets(static_cast<std::size_t>(mp.nprocs) + 1, 0);
    for (int r = 0; r < mp.nprocs; ++r) {
        const std::size_t ur = static_cast<std::size_t>(r);
        rank_offsets[ur + 1] = rank_offsets[ur] + static_cast<cfd::GlobalIndex>(all_n_own[ur]);
    }

    // -------------------------------------------------------------------------
    // 2. Exchange owned local IDs with neighbours to resolve donor-local IDs
    // -------------------------------------------------------------------------
    const int n_nb = mp.n_neighbors();
    std::vector<cfd::LocalIndex> donor_local_ids(mp.recv_ghost_local.size(), 0);
    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(n_nb * 2));

    // Post receives: receive donor's local cell indices into donor_local_ids
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

    // Post sends: send this rank's send_owned_local indices to neighbours
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

    // -------------------------------------------------------------------------
    // 3. Compute linalg GlobalIndex for each ghost: rank_offsets[donor] + local_id
    // -------------------------------------------------------------------------
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

    if (rank == 0) {
        std::fprintf(stderr, "Total test execution time = %.5f sec\n", MPI_Wtime() - t0);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
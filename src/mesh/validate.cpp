#include "cfd/mesh/validate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>
#include <string>

#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::mesh {

namespace {

constexpr LocalIndex kInvalidLocal = static_cast<LocalIndex>(-1);

} // anonymous namespace

void validate_and_log_meshpart(const MeshPart& mp) {
    const int rank = mp.rank;
    const int nprocs = mp.nprocs;
    const MPI_Comm comm = MPI_COMM_WORLD;

    std::stringstream err;
    bool local_ok = true;

    // -------------------------------------------------------------------------
    // 1. Check Fundamental Counts & Array Dimensions
    // -------------------------------------------------------------------------
    if (mp.n_own < 0 || mp.n_own > mp.n_cells) {
        err << "Invalid cell count invariant: n_own (" << mp.n_own << ") > n_cells (" << mp.n_cells << ")\n";
        local_ok = false;
    }
    if (mp.n_nodes_own < 0 || mp.n_nodes_own > mp.n_nodes) {
        err << "Invalid node count invariant: n_nodes_own (" << mp.n_nodes_own << ") > n_nodes (" << mp.n_nodes << ")\n";
        local_ok = false;
    }

    const auto n_cells_sz = static_cast<std::size_t>(mp.n_cells);
    const auto n_faces_sz = static_cast<std::size_t>(mp.n_faces);

    if (mp.cell_type.size() != n_cells_sz ||
        mp.cell_gid.size() != n_cells_sz ||
        mp.cell_donor.size() != n_cells_sz ||
        mp.cell_volume.size() != n_cells_sz ||
        mp.cell_nodes_offsets.size() != n_cells_sz + 1) {
        err << "Cell SoA arrays size mismatch\n";
        local_ok = false;
    }

    if (mp.face_owner.size() != n_faces_sz ||
        mp.face_neigh.size() != n_faces_sz ||
        mp.face_patch.size() != n_faces_sz ||
        mp.face_type.size() != n_faces_sz ||
        mp.face_area.size() != n_faces_sz ||
        mp.face_normal_x.size() != n_faces_sz ||
        mp.face_nodes_offsets.size() != n_faces_sz + 1) {
        err << "Face SoA arrays size mismatch\n";
        local_ok = false;
    }

    // -------------------------------------------------------------------------
    // 2. Validate Cell Topologies and Node Index Ranges
    // -------------------------------------------------------------------------
    for (LocalIndex c = 0; c < mp.n_cells && local_ok; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        const LocalIndex off_start = mp.cell_nodes_offsets[c_sz];
        const LocalIndex off_end   = mp.cell_nodes_offsets[c_sz + 1];
        const auto nnodes = static_cast<int>(off_end - off_start);
        const CellType type = mp.cell_type[c_sz];

        if (nnodes != kNodesPerType[static_cast<std::size_t>(type)]) {
            err << "Cell " << c << " type node count mismatch. Expected: "
                << kNodesPerType[static_cast<std::size_t>(type)] << ", actual: " << nnodes << "\n";
            local_ok = false;
            break;
        }

        for (LocalIndex k = off_start; k < off_end; ++k) {
            const LocalIndex nid = mp.cell_nodes[static_cast<std::size_t>(k)];
            if (nid < 0 || nid >= mp.n_nodes) {
                err << "Cell " << c << " references out-of-range node index " << nid << "\n";
                local_ok = false;
                break;
            }
            if (c < mp.n_own && nid >= mp.n_nodes_own) {
                err << "Owned cell " << c << " references ghost-exclusive node index " << nid << "\n";
                local_ok = false;
                break;
            }
        }

        if (mp.cell_volume[c_sz] <= 1e-15) {
            err << "Degenerate volume in cell " << c << ": " << mp.cell_volume[c_sz] << "\n";
            local_ok = false;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // 3. Validate Face Ordering & Geometric Normal Contract
    // -------------------------------------------------------------------------
    for (LocalIndex f = 0; f < mp.n_faces && local_ok; ++f) {
        const auto f_sz = static_cast<std::size_t>(f);
        const LocalIndex u = mp.face_owner[f_sz];
        const LocalIndex v = mp.face_neigh[f_sz];

        if (u < 0 || u >= mp.n_own) {
            err << "Face " << f << " owner " << u << " is not an owned cell\n";
            local_ok = false;
            break;
        }
        if (v != kInvalidLocal && (v < 0 || v >= mp.n_cells)) {
            err << "Face " << f << " neighbor " << v << " is out of range\n";
            local_ok = false;
            break;
        }

        // Monotonic sorting check: (owner, neigh)
        if (f + 1 < mp.n_faces) {
            const auto next_sz = static_cast<std::size_t>(f + 1);
            const LocalIndex u_next = mp.face_owner[next_sz];
            const LocalIndex v_next = mp.face_neigh[next_sz];
            if (u > u_next || (u == u_next && v > v_next)) {
                err << "Faces are not sorted by (owner, neigh) at index " << f << "\n";
                local_ok = false;
                break;
            }
        }

        // Unit normal assertion
        const double nx = mp.face_normal_x[f_sz];
        const double ny = mp.face_normal_y[f_sz];
        const double nz = mp.face_normal_z[f_sz];
        const double n_len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (std::abs(n_len - 1.0) > 1e-6) {
            err << "Face " << f << " normal vector is not normalized (length = " << n_len << ")\n";
            local_ok = false;
            break;
        }

        // Normal direction: must point outward from owner
        const auto u_sz = static_cast<std::size_t>(u);
        const double dot = nx * (mp.face_centroid_x[f_sz] - mp.cell_centroid_x[u_sz]) +
                           ny * (mp.face_centroid_y[f_sz] - mp.cell_centroid_y[u_sz]) +
                           nz * (mp.face_centroid_z[f_sz] - mp.cell_centroid_z[u_sz]);
        if (dot <= 0.0) {
            err << "Face " << f << " normal orientation contract violated (dot = " << dot << " <= 0)\n";
            local_ok = false;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // 4. Validate BC Patches CSR Mapping
    // -------------------------------------------------------------------------
    const auto n_patches_sz = mp.patches.size();
    if (mp.patch_face_offsets.size() != n_patches_sz + 1 ||
        mp.patch_faces.size() != static_cast<std::size_t>(mp.patch_face_offsets.back())) {
        err << "BC patch offsets/faces size mismatch\n";
        local_ok = false;
    }

    for (std::size_t p = 0; p < n_patches_sz && local_ok; ++p) {
        const LocalIndex off_start = mp.patch_face_offsets[p];
        const LocalIndex off_end   = mp.patch_face_offsets[p + 1];

        for (LocalIndex k = off_start; k < off_end; ++k) {
            const LocalIndex f = mp.patch_faces[static_cast<std::size_t>(k)];
            if (f < 0 || f >= mp.n_faces) {
                err << "Patch " << p << " references out-of-range face " << f << "\n";
                local_ok = false;
                break;
            }
            if (mp.face_patch[static_cast<std::size_t>(f)] != static_cast<PatchId>(p)) {
                err << "Patch " << p << " face " << f << " patch ID mismatch\n";
                local_ok = false;
                break;
            }
            if (mp.face_neigh[static_cast<std::size_t>(f)] != kInvalidLocal) {
                err << "Boundary face " << f << " in patch " << p << " has non-null neighbor cell\n";
                local_ok = false;
                break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. Cross-Rank MPI Halo Communication Symmetry Verification
    // -------------------------------------------------------------------------
    const auto n_nb = mp.n_neighbors();
    std::vector<int> send_counts_to_nb(static_cast<std::size_t>(n_nb));
    std::vector<int> recv_counts_from_nb(static_cast<std::size_t>(n_nb));

    for (int i = 0; i < n_nb; ++i) {
        const auto i_sz = static_cast<std::size_t>(i);
        send_counts_to_nb[i_sz] = static_cast<int>(mp.send_offsets[i_sz + 1] - mp.send_offsets[i_sz]);
        recv_counts_from_nb[i_sz] = static_cast<int>(mp.recv_offsets[i_sz + 1] - mp.recv_offsets[i_sz]);
    }

    // Pairwise verify send/recv volume symmetry with neighbors
    for (int i = 0; i < n_nb && local_ok; ++i) {
        const int target_rank = mp.nb_ranks[static_cast<std::size_t>(i)];
        const int my_send_count = send_counts_to_nb[static_cast<std::size_t>(i)];
        int remote_expected_recv = -1;

        MPI_Sendrecv(&my_send_count, 1, MPI_INT, target_rank, 909,
                     &remote_expected_recv, 1, MPI_INT, target_rank, 909,
                     comm, MPI_STATUS_IGNORE);

        if (recv_counts_from_nb[static_cast<std::size_t>(i)] != remote_expected_recv) {
            err << "MPI Halo Symmetry Mismatch with neighbor Rank " << target_rank
                << ": Local expects to receive " << recv_counts_from_nb[static_cast<std::size_t>(i)]
                << " cells, but Rank " << target_rank << " will send " << remote_expected_recv << "\n";
            local_ok = false;
        }
    }

    // Global abort on any failure
    int global_ok_int = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok_int, 1, MPI_INT, MPI_MIN, comm);

    if (global_ok_int == 0) {
        if (!local_ok) {
            std::stringstream ss;
            ss << "Sanity Check FAILED on Rank " << rank << ":\n" << err.str();
            std::string result = ss.str();
            mpi::fatal(comm, result);
        } else {
            mpi::fatal(comm, "Sanity Check FAILED on remote ranks.");
        }
    }

    // -------------------------------------------------------------------------
    // 6. Collective Global Diagnostics Logging (Rank 0)
    // -------------------------------------------------------------------------
    std::uint64_t loc_owned_cells = static_cast<std::uint64_t>(mp.n_own);
    std::uint64_t loc_ghost_cells = static_cast<std::uint64_t>(mp.n_cells - mp.n_own);
    std::uint64_t loc_owned_nodes = static_cast<std::uint64_t>(mp.n_nodes_own);

    std::uint64_t min_cells = 0, max_cells = 0;
    std::uint64_t min_ghosts = 0, max_ghosts = 0;
    std::uint64_t total_cells = 0, total_nodes = 0;

    MPI_Reduce(&loc_owned_cells, &min_cells, 1, MPI_UINT64_T, MPI_MIN, 0, comm);
    MPI_Reduce(&loc_owned_cells, &max_cells, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
    MPI_Reduce(&loc_owned_cells, &total_cells, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    MPI_Reduce(&loc_ghost_cells, &min_ghosts, 1, MPI_UINT64_T, MPI_MIN, 0, comm);
    MPI_Reduce(&loc_ghost_cells, &max_ghosts, 1, MPI_UINT64_T, MPI_MAX, 0, comm);

    MPI_Reduce(&loc_owned_nodes, &total_nodes, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    double loc_vol = 0.0;
    for (LocalIndex c = 0; c < mp.n_own; ++c) loc_vol += mp.cell_volume[static_cast<std::size_t>(c)];
    double total_domain_vol = 0.0;
    MPI_Reduce(&loc_vol, &total_domain_vol, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        const double avg_cells = static_cast<double>(total_cells) / static_cast<double>(nprocs);
        const double imbalance = (avg_cells > 0.0) ? (static_cast<double>(max_cells) / avg_cells - 1.0) * 100.0 : 0.0;

        mpi::log_stat("===============================================================================");
        mpi::log_stat("                       FINAL MESH QUALITY & SANITY AUDIT                       ");
        mpi::log_stat("===============================================================================");
        mpi::log_stat(" [OK] All invariants, face orientations, and communication graphs verified.");
        mpi::log_stat(" Global Topology Summary:");
        mpi::log_stat("   - Total Volume Cells   : %llu", static_cast<unsigned long long>(total_cells));
        mpi::log_stat("   - Total Unique Nodes   : %llu", static_cast<unsigned long long>(mp.n_nodes_g));
        mpi::log_stat("   - Total Unique Faces   : %llu", static_cast<unsigned long long>(mp.n_faces_g));
        mpi::log_stat("   - Total Boundary Faces : %llu", static_cast<unsigned long long>(mp.n_bfaces_g));
        mpi::log_stat("   - Total Domain Volume  : %.10e", total_domain_vol);
        mpi::log_stat(" Domain Partitioning & Communication:");
        mpi::log_stat("   - Owned Cells / Rank   : min = %llu, avg = %.1f, max = %llu (Imbalance: %.2f%%)",
                      static_cast<unsigned long long>(min_cells), avg_cells,
                      static_cast<unsigned long long>(max_cells), imbalance);
        mpi::log_stat("   - Ghost Cells / Rank   : min = %llu, max = %llu",
                      static_cast<unsigned long long>(min_ghosts), static_cast<unsigned long long>(max_ghosts));
        mpi::log_stat("   - Boundary Patches     : %zu", n_patches_sz);
        mpi::log_stat("===============================================================================");
    }
}

} // namespace cfd::mesh
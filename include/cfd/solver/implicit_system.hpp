#pragma once

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "cfd/core/types.hpp"
//#include "cfd/linalg/config.hpp"
#include "cfd/linalg/config.hpp"
#include "cfd/linalg/preconditioners.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/linalg/solver.hpp"
#include "cfd/linalg/vector.hpp"
#include "cfd/linalg/preconditioner.hpp"
#include "cfd/linalg/bsr_matrix.hpp"
#include "cfd/linalg/bicgstab.hpp"



namespace cfd::solver {

using linalg::detail::mpi_index_type;

/**
 * @class MeanFlowSystem
 * @brief Builds, fills and solves the distributed 5x5 block system of one
 *        implicit step for the mean flow.
 */
class MeanFlowSystem {
public:
    static constexpr std::size_t kNumVars = constants::kNumVars;
    static constexpr std::size_t kBlockSize = static_cast<std::size_t>(kNumVars) * static_cast<std::size_t>(kNumVars);
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::CellCellsByFace;   // dual graph

    /**
     * @param mesh              Rank-local mesh (faces, ghost maps, global ids).
     * @param aux_conn          Auxiliary mesh connectivities
     * @param comm              Solver communicator.
     */
    MeanFlowSystem(const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn, MPI_Comm comm)
        : mesh_(mesh), aux_conn_(aux_conn), comm_(comm), n_own_(static_cast<std::size_t>(mesh.n_own)) {
        matrix_ = std::make_unique<linalg::BsrMatrix>(comm_, mesh_.n_cells_g, n_own_, kNumVars);

        const std::vector<cfd::GlobalIndex> ghost_gids = build_ghost_gids(mesh_, comm);
        matrix_->assemble(ghost_gids, aux_conn_.cell_cells_face_offsets, aux_conn_.cell_cells_face);
 
        rhs_ = matrix_->makeVector();
        du_ = matrix_->makeVector();
        
        // solver_params_ = linalg::parse_solver_config(const std::string &path, const MPI_Comm comm)    // TBD //

        if (solver_params_.type == linalg::SolverType::BICGSTAB) {
            solver_ = std::make_unique<linalg::BiCGSTAB>();
            solver_->params() = solver_params_;
        }

        if (solver_->params().precond_type == linalg::PreconditionerType::None) {
            preconditioner_ = std::make_unique<linalg::IdentityPreconditioner>();

        } else if (solver_->params().precond_type == linalg::PreconditionerType::SGS) {
            preconditioner_ = std::make_unique<linalg::SgsPreconditioner>();
            // set sweeps // TBD //
        }

        built_ = true;
    }

    bool solve() {
        preconditioner_->setup(*matrix_);
        last_ = solver_->solve(*matrix_, *preconditioner_, du_, rhs_);
        return last_.status == linalg::SolverStatus::Converged;
    }

    linalg::BsrMatrix& matrix() noexcept { return *matrix_; }
    const linalg::BsrMatrix& matrix() const noexcept { return *matrix_; }

    double* rhs_data() noexcept { return rhs_.data(); }  
    const double* rhs_data() const noexcept { return rhs_.data(); }  
    double* du_data() noexcept { return du_.data(); }  
    const double* du_data() const noexcept { return du_.data(); }  

    std::size_t nown() const noexcept { return n_own_; }

private:
    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    MPI_Comm comm_;
    std::size_t n_own_ = 0;
    bool built_ = false;

    linalg::SolverParams solver_params_;
    std::unique_ptr<linalg::IterativeSolver> solver_;
    std::unique_ptr<linalg::BsrMatrix> matrix_;
    std::unique_ptr<linalg::Preconditioner> preconditioner_;
    linalg::Vector rhs_, du_;
    linalg::IterationResult last_{};

    // make map: cfd ghost ids -> linalg ghost ids
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

};

}  // namespace cfd::solver::implicit

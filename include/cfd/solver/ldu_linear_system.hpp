#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/linalg/config.hpp"
#include "cfd/linalg/preconditioners.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/linalg/solver.hpp"
#include "cfd/linalg/vector.hpp"
#include "cfd/linalg/preconditioner.hpp"
#include "cfd/linalg/ldu_matrix.hpp"
#include "cfd/linalg/bicgstab.hpp"

namespace cfd::solver {

/**
 * @class LduLinearSystem
 * @brief Distributed scalar LDU linear system manager (A * x_c = b_c).
 * 
 * Supports both single scalar variables (NumComponents = 1, e.g. Pressure Poisson)
 * and multi-component vector fields sharing a single matrix operator
 * (NumComponents = 3, e.g. Momentum equations u, v, w).
 *
 * @tparam NumComponents Number of RHS and solution vectors sharing the operator A.
 */
template <std::size_t NumComponents = 1>
class LduLinearSystem {
public:
    static constexpr std::size_t kNumComponents = NumComponents;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::CellCellsByFace;

    /**
     * @param mesh          Local partitioned mesh.
     * @param aux_conn      Cell-to-cell dual connectivity graph.
     * @param solver_params Krylov solver and preconditioner configuration.
     * @param comm          MPI communicator.
     */
    LduLinearSystem(const mesh::MeshPart& mesh,
                    const mesh::MeshAuxConnectivity& aux_conn,
                    const linalg::SolverParams& solver_params,
                    MPI_Comm comm)
        : mesh_(mesh),
          aux_conn_(aux_conn),
          solver_params_(solver_params),
          comm_(comm),
          n_own_(static_cast<std::size_t>(mesh.n_own)) {

        matrix_ = std::make_unique<linalg::LduMatrix>(comm_, mesh_.n_cells_g, n_own_);

        const std::vector<GlobalIndex> ghost_gids = build_ghost_gids(mesh_, comm_);
        matrix_->assemble(ghost_gids, aux_conn_.cell_cells_face_offsets, aux_conn_.cell_cells_face);

        for (std::size_t c = 0; c < NumComponents; ++c) {
            rhs_[c] = matrix_->makeVector();
            sol_[c] = matrix_->makeVector();
        }

        if (solver_params_.type == linalg::SolverType::BICGSTAB) {
            solver_ = std::make_unique<linalg::BiCGSTAB>();
            solver_->params() = solver_params_;
        }

        if (solver_->params().precond_type == linalg::PreconditionerType::None) {
            preconditioner_ = std::make_unique<linalg::IdentityPreconditioner>();
        } else if (solver_->params().precond_type == linalg::PreconditionerType::SGS) {
            preconditioner_ = std::make_unique<linalg::SgsPreconditioner>();
        }

        built_ = true;
    }

    /**
     * @brief Zero-out matrix entries (diag, upper, lower) and all RHS/solution vectors.
     */
    void zero() noexcept {
        matrix_->setZero();
        for (std::size_t c = 0; c < NumComponents; ++c) {
            rhs_[c].setZero();
            sol_[c].setZero();
        }
    }

    /**
     * @brief Solves the linear system(s).
     * Preconditioner is factored ONCE and reused across all components.
     * 
     * @return true if ALL components converged within tolerance.
     */
    bool solve() {
        preconditioner_->setup(*matrix_);

        bool all_converged = true;
        for (std::size_t c = 0; c < NumComponents; ++c) {
            last_results_[c] = solver_->solve(*matrix_, *preconditioner_, sol_[c], rhs_[c]);
            if (last_results_[c].status != linalg::SolverStatus::Converged) {
                all_converged = false;
            }
        }
        return all_converged;
    }

    [[nodiscard]] linalg::LduMatrix& matrix() noexcept { return *matrix_; }
    [[nodiscard]] const linalg::LduMatrix& matrix() const noexcept { return *matrix_; }

    [[nodiscard]] double* rhs_data(std::size_t comp = 0) noexcept { 
        assert(comp < NumComponents);
        return rhs_[comp].data(); 
    }
    [[nodiscard]] const double* rhs_data(std::size_t comp = 0) const noexcept { 
        assert(comp < NumComponents);
        return rhs_[comp].data(); 
    }

    [[nodiscard]] double* sol_data(std::size_t comp = 0) noexcept { 
        assert(comp < NumComponents);
        return sol_[comp].data(); 
    }
    [[nodiscard]] const double* sol_data(std::size_t comp = 0) const noexcept { 
        assert(comp < NumComponents);
        return sol_[comp].data(); 
    }

    [[nodiscard]] linalg::Vector& rhs_vector(std::size_t comp = 0) noexcept { return rhs_[comp]; }
    [[nodiscard]] linalg::Vector& sol_vector(std::size_t comp = 0) noexcept { return sol_[comp]; }

    [[nodiscard]] double* rhs_u_data() noexcept requires (NumComponents == 3) { return rhs_[0].data(); }
    [[nodiscard]] double* rhs_v_data() noexcept requires (NumComponents == 3) { return rhs_[1].data(); }
    [[nodiscard]] double* rhs_w_data() noexcept requires (NumComponents == 3) { return rhs_[2].data(); }

    [[nodiscard]] double* sol_u_data() noexcept requires (NumComponents == 3) { return sol_[0].data(); }
    [[nodiscard]] double* sol_v_data() noexcept requires (NumComponents == 3) { return sol_[1].data(); }
    [[nodiscard]] double* sol_w_data() noexcept requires (NumComponents == 3) { return sol_[2].data(); }

    [[nodiscard]] std::size_t n_own() const noexcept { return n_own_; }
    [[nodiscard]] const linalg::IterationResult& last_result(std::size_t comp = 0) const noexcept {
        assert(comp < NumComponents);
        return last_results_[comp];
    }

private:
    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    const linalg::SolverParams& solver_params_;
    MPI_Comm comm_;
    std::size_t n_own_ = 0;
    bool built_ = false;

    std::unique_ptr<linalg::IterativeSolver> solver_;
    std::unique_ptr<linalg::LduMatrix> matrix_;
    std::unique_ptr<linalg::Preconditioner> preconditioner_;

    std::array<linalg::Vector, NumComponents> rhs_;
    std::array<linalg::Vector, NumComponents> sol_;
    std::array<linalg::IterationResult, NumComponents> last_results_{};

    std::vector<GlobalIndex> build_ghost_gids(const mesh::MeshPart& mp, MPI_Comm comm) {
        const MPI_Datatype mpi_local_type = (sizeof(LocalIndex) == 8) ? MPI_INT64_T : MPI_INT32_T;

        std::vector<LocalIndex> all_n_own(static_cast<std::size_t>(mp.nprocs), 0);
        MPI_Allgather(&mp.n_own, 1, mpi_local_type,
                      all_n_own.data(), 1, mpi_local_type, comm);

        std::vector<GlobalIndex> rank_offsets(static_cast<std::size_t>(mp.nprocs) + 1, 0);
        for (int r = 0; r < mp.nprocs; ++r) {
            const std::size_t ur = static_cast<std::size_t>(r);
            rank_offsets[ur + 1] = rank_offsets[ur] + static_cast<GlobalIndex>(all_n_own[ur]);
        }

        const int n_nb = mp.n_neighbors();
        std::vector<LocalIndex> donor_local_ids(mp.recv_ghost_local.size(), 0);
        std::vector<MPI_Request> reqs;
        reqs.reserve(static_cast<std::size_t>(n_nb * 2));

        for (int k = 0; k < n_nb; ++k) {
            const std::size_t uk = static_cast<std::size_t>(k);
            const int neighbor = mp.nb_ranks[uk];
            const LocalIndex offset = mp.recv_offsets[uk];
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
            const LocalIndex offset = mp.send_offsets[uk];
            const int count = static_cast<int>(mp.send_offsets[uk + 1] - offset);
            if (count > 0) {
                MPI_Request req = MPI_REQUEST_NULL;
                MPI_Isend(const_cast<LocalIndex*>(mp.send_owned_local.data() + offset),
                          count, mpi_local_type, neighbor, 4318, comm, &req);
                reqs.push_back(req);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        std::vector<GlobalIndex> ghost_gids;
        ghost_gids.reserve(mp.recv_ghost_local.size());

        for (int k = 0; k < n_nb; ++k) {
            const std::size_t uk = static_cast<std::size_t>(k);
            const int donor_rank = mp.nb_ranks[uk];
            const GlobalIndex donor_offset = rank_offsets[static_cast<std::size_t>(donor_rank)];
            const LocalIndex begin_idx = mp.recv_offsets[uk];
            const LocalIndex end_idx = mp.recv_offsets[uk + 1];

            for (LocalIndex i = begin_idx; i < end_idx; ++i) {
                const LocalIndex donor_local = donor_local_ids[static_cast<std::size_t>(i)];
                ghost_gids.push_back(donor_offset + static_cast<GlobalIndex>(donor_local));
            }
        }

        std::sort(ghost_gids.begin(), ghost_gids.end());
        ghost_gids.erase(std::unique(ghost_gids.begin(), ghost_gids.end()), ghost_gids.end());
        return ghost_gids;
    }
};

} // namespace cfd::solver
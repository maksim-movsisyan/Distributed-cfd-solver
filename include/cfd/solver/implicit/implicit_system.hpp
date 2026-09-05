// Implicit mean-flow linear system: (V/dt + dR/dU) du = -R.
//
// Native matrix initialization from the mesh: the BSR structure (one 5x5
// block per cell pair coupled by a face) is derived ONCE from the face
// adjacency, together with a per-row lookup (sorted neighbour cell ids +
// binary search) that maps (row cell, column cell) -> storage slot. Every
// subsequent time step fills the values in place through that lookup — no COO
// staging, no intermediate matrices, no reassembly.
//
// System numbering: linear-system rows are a contiguous renumbering of the
// OWNED cells (row r <-> owned cell r; global row id = row_begin_ + r via
// prefix sum). Columns referencing neighbour ranks' cells become ghost
// columns of the linalg matrix — the linear solver's own halo machinery then
// moves solution increments between ranks, so this class does no field
// communication beyond a one-time structure setup.
//
// STATUS (work in progress): on strongly anisotropic supersonic cases the
// assembled system defeats the SGS-preconditioned BiCGSTAB within a practical
// iteration budget. Two properties of the raw system drive this: conserved-
// variable Jacobians carry entries dominating both their row and column
// diagonals by orders of magnitude (e.g. dF_E/drho at fixed momentum and
// energy), and (supersonic) flux Jacobians have nearly vanishing row sums.
// enforce_diagonal_dominance() applies the standard Blazek-style matrix
// modification; remaining candidates are ILU(0), better variable scaling, or
// an analytic Jacobian expressed directly in primitive variables.
#pragma once

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/linalg/linalg.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::solver::implicit {

using linalg::detail::mpi_index_type;

/**
 * @class MeanFlowSystem
 * @brief Builds, fills and solves the distributed 5x5 block system of one
 *        implicit step for the mean flow.
 */
class MeanFlowSystem {
public:
    static constexpr int kNumVars = constants::kNumVars;

    /**
     * @param mp              Rank-local mesh (faces, ghost maps, global ids).
     * @param comm            Solver communicator (collectives in build()).
     * @param linear_rtol     BiCGSTAB relative residual tolerance per step.
     * @param linear_max_iter BiCGSTAB iteration budget per step.
     */
    MeanFlowSystem(const mesh::MeshPart& mp, MPI_Comm comm,
                   const double linear_rtol, const int linear_max_iter)
        : mp_(mp), comm_(comm), n_own_(static_cast<std::size_t>(mp.n_own)) {
        solver_.params().relative_tolerance = linear_rtol;
        solver_.params().max_iterations = linear_max_iter;
        solver_.params().verbosity =
            g_verbose >= 2 ? linalg::Verbosity::Verbose : linalg::Verbosity::Silent;
        solver_.params().verify_final_residual = false;
    }

    /**
     * @brief One-time structure construction (collective on comm).
     *
     * Derives the sparsity from the face adjacency, resolves the system row
     * ids of MPI ghost cells with their donor ranks, assembles the linalg
     * matrix structure and builds the per-row slot lookup.
     */
    void build() {
        if (built_) return;
        compute_row_offsets();
        exchange_ghost_rows();
        build_matrix_structure();
        build_slot_lookup();
        rhs_ = A_->makeVector();
        du_ = A_->makeVector();
        built_ = true;
    }

    bool built() const noexcept { return built_; }

    // --- assembly (native in-place fills; call after build()) -----------------

    /// Zeroes all blocks — start of a time step's assembly.
    void begin_assembly() {
        double* CFD_RESTRICT v = A_->valuesData();
        std::fill(v, v + A_->values().size(), 0.0);
    }

    /// Diagonal time term: J(cell, cell) += (volume / dt) * I.
    void add_time_term(LocalIndex cell, double volume_over_dt) {
        const std::size_t slot = diag_slot(cell);
        double* CFD_RESTRICT b = A_->valuesData() + slot * kBlockSize;
        for (int i = 0; i < kNumVars; ++i) {
            b[i * kNumVars + i] += volume_over_dt;
        }
    }

    /**
     * @brief J(row_cell, col_cell) += scale * block (5x5, row-major).
     *
     * Rows outside the owned partition (ghost cells — their rows live on the
     * donor rank) are dropped; that is exactly the ownership split of the
     * distributed assembly.
     */
    void add_block(LocalIndex row_cell, LocalIndex col_cell, double scale,
                   const double* block) {
        if (static_cast<std::size_t>(row_cell) >= n_own_) return;
        const std::size_t slot = find_slot(row_cell, col_cell);
        double* CFD_RESTRICT dst = A_->valuesData() + slot * kBlockSize;
        for (std::size_t i = 0; i < kBlockSize; ++i) {
            dst[i] += scale * block[i];
        }
    }

    /** @brief RHS from the residual slots: b = -R, owned cells. */
    void set_rhs_from_residual(std::span<double* const> res_slots) {
        double* CFD_RESTRICT b = rhs_.data();
        for (std::size_t c = 0; c < n_own_; ++c) {
            for (int v = 0; v < kNumVars; ++v) {
                b[c * kNumVars + v] = -res_slots[v][c];
            }
        }
    }

    // --- solve ------------------------------------------------------------------

    /** @brief Applies the dominance modification, refactors and solves. */
    void solve() {
        enforce_diagonal_dominance();
        sgs_.setup(*A_);
        du_.setZero();
        last_ = solver_.solve(*A_, sgs_, du_, rhs_);
    }

    /** @brief Solution increment: du[cell * 5 + var], owned cells. */
    const double* increment() const noexcept { return du_.data(); }

    int linear_iterations() const noexcept { return last_.iterations; }
    bool linear_converged() const noexcept {
        return last_.status == linalg::SolverStatus::Converged;
    }

private:
    static constexpr std::size_t kBlockSize =
        static_cast<std::size_t>(kNumVars) * static_cast<std::size_t>(kNumVars);

    // --- build steps -------------------------------------------------------------

    // Contiguous system rows: rank r owns rows [begin_r, begin_r + n_own_r).
    void compute_row_offsets() {
        int rank = 0, nprocs = 1;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &nprocs);
        const GlobalIndex n_local = static_cast<GlobalIndex>(mp_.n_own);
        MPI_Exscan(&n_local, &row_begin_, 1, mpi_index_type(), MPI_SUM, comm_);
        if (rank == 0) row_begin_ = 0;

        all_row_begins_.assign(static_cast<std::size_t>(nprocs), 0);
        MPI_Allgather(&row_begin_, 1, mpi_index_type(), all_row_begins_.data(), 1,
                      mpi_index_type(), comm_);
    }

    // Resolves the global system row id of every MPI ghost cell: ask the donor
    // rank for its local index of that global cell (two-phase exchange).
    void exchange_ghost_rows() {
        const std::size_t n_cells = static_cast<std::size_t>(mp_.n_cells);
        int nprocs = 1;
        MPI_Comm_size(comm_, &nprocs);
        const std::size_t p = static_cast<std::size_t>(nprocs);

        // Requests: my ghost-cell global ids grouped by donor rank.
        std::vector<int> send_counts(p, 0);
        for (std::size_t c = n_own_; c < n_cells; ++c) {
            ++send_counts[static_cast<std::size_t>(mp_.cell_donor[c])];
        }
        std::vector<int> recv_counts(p, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);

        auto displs = [](const std::vector<int>& counts) {
            std::vector<int> d(counts.size() + 1, 0);
            for (std::size_t i = 0; i < counts.size(); ++i) d[i + 1] = d[i] + counts[i];
            return d;
        };
        const std::vector<int> sdispls = displs(send_counts);
        const std::vector<int> rdispls = displs(recv_counts);

        std::vector<GlobalIndex> request_ids(static_cast<std::size_t>(sdispls.back()));
        std::vector<int> send_cursor(sdispls.begin(), sdispls.end() - 1);
        for (std::size_t c = n_own_; c < n_cells; ++c) {
            const auto d = static_cast<std::size_t>(mp_.cell_donor[c]);
            request_ids[static_cast<std::size_t>(send_cursor[d]++)] = mp_.cell_gid[c];
        }

        std::vector<GlobalIndex> incoming(static_cast<std::size_t>(rdispls.back()));
        MPI_Alltoallv(request_ids.data(), send_counts.data(), sdispls.data(),
                      mpi_index_type(), incoming.data(), recv_counts.data(),
                      rdispls.data(), mpi_index_type(), comm_);

        // Answer with the local owned index of each requested global cell.
        std::unordered_map<GlobalIndex, LocalIndex> owned_of_gid;
        owned_of_gid.reserve(2 * n_own_);
        for (std::size_t c = 0; c < n_own_; ++c) {
            owned_of_gid.emplace(mp_.cell_gid[c], static_cast<LocalIndex>(c));
        }
        std::vector<LocalIndex> answers(static_cast<std::size_t>(rdispls.back()));
        for (std::size_t i = 0; i < incoming.size(); ++i) {
            const auto it = owned_of_gid.find(incoming[i]);
            if (it == owned_of_gid.end()) {
                linalg::fatal(comm_, "implicit: ghost cell global id not owned by its donor rank");
            }
            answers[i] = it->second;
        }

        std::vector<LocalIndex> resolved(static_cast<std::size_t>(sdispls.back()));
        MPI_Alltoallv(answers.data(), recv_counts.data(), rdispls.data(), mpi_index_type(),
                      resolved.data(), send_counts.data(), sdispls.data(), mpi_index_type(),
                      comm_);

        ghost_system_row_.assign(n_cells - n_own_, 0);
        std::vector<int> recv_cursor(sdispls.begin(), sdispls.end() - 1);
        for (std::size_t c = n_own_; c < n_cells; ++c) {
            const auto d = static_cast<std::size_t>(mp_.cell_donor[c]);
            const LocalIndex local = resolved[static_cast<std::size_t>(recv_cursor[d]++)];
            ghost_system_row_[c - n_own_] = all_row_begins_[d] + local;
        }
    }

    // Global system column id of a mesh cell.
    [[nodiscard]] GlobalIndex system_col(LocalIndex cell) const {
        if (static_cast<std::size_t>(cell) < n_own_) {
            return row_begin_ + static_cast<GlobalIndex>(cell);
        }
        return ghost_system_row_[static_cast<std::size_t>(cell) - n_own_];
    }

    // BSR structure from the face adjacency: block (c0, c1) for every inner
    // face pair (both directions) plus the diagonal.
    void build_matrix_structure() {
        A_ = std::make_unique<linalg::BsrMatrix>(comm_, mp_.n_cells_g, mp_.n_own, kNumVars);

        const std::size_t n_inner = static_cast<std::size_t>(mp_.n_inner_faces);
        const LocalIndex* CFD_RESTRICT owner = mp_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mp_.face_neigh.data();

        // Reserve exactly: diagonal + two directed blocks per inner face.
        const std::size_t n_blocks = n_own_ + 2 * n_inner;
        A_->reserve(n_blocks);

        const double zero_block[kBlockSize] = {};
        for (std::size_t c = 0; c < n_own_; ++c) {
            A_->addBlock(row_begin_ + static_cast<GlobalIndex>(c),
                         row_begin_ + static_cast<GlobalIndex>(c), zero_block);
        }
        for (std::size_t f = 0; f < n_inner; ++f) {
            const LocalIndex c0 = owner[f];  // always an owned cell
            const LocalIndex c1 = neigh[f];  // owned or MPI ghost
            A_->addBlock(row_begin_ + c0, system_col(c1), zero_block);
            // The reverse block belongs to the owner of c1: staged here only
            // when c1 is owned, otherwise by the donor rank (which sees this
            // face with its own cell as owner and ours as the ghost).
            if (static_cast<std::size_t>(c1) < n_own_) {
                A_->addBlock(row_begin_ + c1, system_col(c0), zero_block);
            }
        }
        A_->assemble();
    }

    // Per-row lookup: for each owned row, the neighbour MESH cell ids (sorted,
    // unique) and the parallel matrix storage slots — binary-search fills.
    void build_slot_lookup() {
        // linalg-local column slot -> mesh cell id (invert the ghost mapping).
        std::unordered_map<GlobalIndex, LocalIndex> cell_of_system;
        cell_of_system.reserve(2 * ghost_system_row_.size());
        for (std::size_t c = n_own_; c < ghost_system_row_.size() + n_own_; ++c) {
            cell_of_system.emplace(ghost_system_row_[c - n_own_], static_cast<LocalIndex>(c));
        }
        const auto& layout_ghosts = A_->layout().ghostGlobalIds();

        const auto& row_ptr = A_->rowPtr();
        const auto& cols = A_->cols();
        row_col_offsets_.assign(n_own_ + 1, 0);
        row_col_cells_.resize(static_cast<std::size_t>(row_ptr[n_own_]));
        row_slots_.resize(static_cast<std::size_t>(row_ptr[n_own_]));

        std::vector<std::pair<LocalIndex, LocalIndex>> row_pairs;
        for (std::size_t r = 0; r < n_own_; ++r) {
            row_pairs.clear();
            for (LocalIndex k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                const LocalIndex s = cols[static_cast<std::size_t>(k)];
                LocalIndex cell;
                if (static_cast<std::size_t>(s) < n_own_) {
                    cell = s;
                } else {
                    const GlobalIndex gid = layout_ghosts[static_cast<std::size_t>(s) - n_own_];
                    cell = cell_of_system.at(gid);
                }
                row_pairs.emplace_back(cell, k);
            }
            std::sort(row_pairs.begin(), row_pairs.end());
            row_col_offsets_[r + 1] = row_col_offsets_[r] +
                                      static_cast<LocalIndex>(row_pairs.size());
            for (std::size_t i = 0; i < row_pairs.size(); ++i) {
                row_col_cells_[static_cast<std::size_t>(row_col_offsets_[r]) + i] =
                    row_pairs[i].first;
                row_slots_[static_cast<std::size_t>(row_col_offsets_[r]) + i] =
                    row_pairs[i].second;
            }
        }
    }

    // --- slot lookup -------------------------------------------------------------

    [[nodiscard]] std::size_t find_slot(LocalIndex row_cell, LocalIndex col_cell) const {
        const auto first =
            row_col_cells_.begin() +
            static_cast<std::ptrdiff_t>(row_col_offsets_[static_cast<std::size_t>(row_cell)]);
        const auto last =
            row_col_cells_.begin() +
            static_cast<std::ptrdiff_t>(row_col_offsets_[static_cast<std::size_t>(row_cell) + 1]);
        const auto it = std::lower_bound(first, last, col_cell);
        if (it == last || *it != col_cell) {
            linalg::fatal(comm_,
                          "implicit: assembly touched a cell pair outside the matrix structure");
        }
        return static_cast<std::size_t>(
            row_slots_[static_cast<std::size_t>(it - row_col_cells_.begin())]);
    }

    [[nodiscard]] std::size_t diag_slot(LocalIndex cell) const {
        return find_slot(cell, cell);
    }

    /**
     * @brief Diagonal-dominance modification of the implicit operator
     *        (Blazek-style Jacobian modification): each diagonal block is
     *        lifted so its row-sum norm meets the sum of its off-diagonal
     *        block norms.
     *
     * Flux Jacobians of (supersonic) flows have nearly vanishing row sums and
     * energy rows dominating both their row and column scales; the raw matrix
     * is far from block-diagonally dominant and Gauss-Seidel type
     * preconditioners diverge on it. This modification is the standard robust
     * fix: the steady state is unaffected (du -> 0 whenever R -> 0, whatever
     * the matrix), only the path to it changes.
     */
    void enforce_diagonal_dominance() {
        constexpr int N = kNumVars;
        double* CFD_RESTRICT av = A_->valuesData();
        const auto& row_ptr = A_->rowPtr();
        const auto& cols = A_->cols();

        for (std::size_t r = 0; r < n_own_; ++r) {
            double diag_norm = 0.0;
            double off_sum = 0.0;
            std::size_t dslot = 0;
            for (LocalIndex k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                const double* b = av + static_cast<std::size_t>(k) * kBlockSize;
                double rn = 0.0;
                for (int i = 0; i < N; ++i) {
                    double s = 0.0;
                    for (int j = 0; j < N; ++j) {
                        s += std::fabs(b[i * N + j]);
                    }
                    rn = std::max(rn, s);
                }
                if (cols[static_cast<std::size_t>(k)] == static_cast<LocalIndex>(r)) {
                    diag_norm = rn;
                    dslot = static_cast<std::size_t>(k);
                } else {
                    off_sum += rn;
                }
            }
            if (off_sum > diag_norm) {
                double* db = av + dslot * kBlockSize;
                const double add = 1.0000001 * (off_sum - diag_norm);
                for (int v = 0; v < N; ++v) {
                    db[v * N + v] += add;
                }
            }
        }
    }

    // --- state -------------------------------------------------------------------

    const mesh::MeshPart& mp_;
    MPI_Comm comm_;
    std::size_t n_own_ = 0;
    bool built_ = false;

    GlobalIndex row_begin_ = 0;
    std::vector<GlobalIndex> all_row_begins_;
    std::vector<GlobalIndex> ghost_system_row_;  // per mesh ghost cell

    std::unique_ptr<linalg::BsrMatrix> A_;
    linalg::SgsPreconditioner sgs_;
    linalg::BiCGSTAB solver_;
    linalg::Vector rhs_, du_;
    linalg::IterationResult last_{};

    std::vector<LocalIndex> row_col_offsets_;  // n_own + 1
    std::vector<LocalIndex> row_col_cells_;    // sorted per row
    std::vector<LocalIndex> row_slots_;        // parallel to row_col_cells_
};

}  // namespace cfd::solver::implicit

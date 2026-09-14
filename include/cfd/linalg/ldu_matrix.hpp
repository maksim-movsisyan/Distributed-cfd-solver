#pragma once

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/linalg/operator.hpp"

namespace cfd::linalg {

/**
 * @class LduMatrix
 * @brief Distributed scalar LDU (Lower-Diagonal-Upper) matrix operator.
 *
 * Designed for segregated incompressible solvers (SIMPLE/PISO):
 *  - Stores main diagonal (owned rows), upper triangle, and lower triangle as flat arrays.
 *  - Supports distributed memory (MPI) via VectorLayout.
 *  - Internal faces have symmetric lower/upper storage.
 *  - Inter-processor interface edges (owned -> ghost) are stored in the upper array,
 *    allowing zero-overhead overlap with halo updates.
 *
 * Implements the LinearOperator interface.
 */
class LduMatrix : public LinearOperator {
public:
    // Collective: contiguous distribution of `n_global_rows` scalar rows across MPI ranks.
    LduMatrix(MPI_Comm comm, GlobalIndex n_global_rows, LocalIndex n_local_rows)
        : layout_(comm, n_global_rows, n_local_rows) {}

    // --- LinearOperator ---
    GlobalIndex globalRows() const override { return layout_.globalSize(); }
    LocalIndex localRows() const override { return layout_.localSize(); }
    int blockSize() const override { return 1; }
    const VectorLayout& layout() const override { return layout_; }
    Vector makeVector() const override { return Vector(layout_, 1); }
    void apply(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;

    // --- Raw access (for kernels, assembly, preconditioners) ---
    const std::vector<double>& diag() const { return diag_; }
    std::vector<double>& diag() { return diag_; }
    double* diagData() { return diag_.data(); }
    const double* diagData() const { return diag_.data(); }

    const std::vector<double>& upper() const { return upper_; }
    std::vector<double>& upper() { return upper_; }
    double* upperData() { return upper_.data(); }
    const double* upperData() const { return upper_.data(); }

    const std::vector<double>& lower() const { return lower_; }
    std::vector<double>& lower() { return lower_; }
    double* lowerData() { return lower_.data(); }
    const double* lowerData() const { return lower_.data(); }

    // --- Graph topology accessors ---
    const std::vector<LocalIndex>& ownerStart() const { return owner_start_; }
    const std::vector<LocalIndex>& owner() const { return owner_; }
    const std::vector<LocalIndex>& neigh() const { return neigh_; }

    std::size_t numInternalEdges() const { return n_internal_edges_; }
    std::size_t numTotalEdges() const { return owner_.size(); }
    bool assembled() const { return assembled_; }

    void setZero() noexcept;

    // --- Graph-based assembling (n_total = own + ghosts) ---
    /**
     * @brief Assembles the LDU sparsity pattern from cell-cell dual graph adjacency.
     */
    void assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                  const std::vector<LocalIndex>& dual_graph_off,
                  const std::vector<LocalIndex>& dual_graph_val);

    /**
     * @brief Assembles directly from mesh face-to-cell connectivity.
     */
    void assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                  const std::vector<LocalIndex>& face_owner,
                  const std::vector<LocalIndex>& face_neigh,
                  std::size_t n_internal_faces);

private:
    VectorLayout layout_;

    // Topology:
    // owner_start_ [localRows + 1]: offsets into internal edges owned by each cell (for SGS)
    std::vector<LocalIndex> owner_start_;
    std::vector<LocalIndex> owner_;       // [numTotalEdges]
    std::vector<LocalIndex> neigh_;       // [numTotalEdges]
    std::size_t n_internal_edges_ = 0;    // Edges with both nodes in [0, localRows)

    // Numerical values:
    std::vector<double> diag_;            // [localRows]
    std::vector<double> upper_;           // [numTotalEdges] (internal + interface)
    std::vector<double> lower_;           // [numInternalEdges] (only internal pairs)

    bool assembled_ = false;
};

} // namespace cfd::linalg
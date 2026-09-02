#pragma once

#include <mpi.h>

#include <memory>
#include <vector>

#include "cfd/core/types.hpp"

namespace cfd::linalg {

/**
 * @class VectorLayout
 * @brief Row distribution + ghost exchange pattern shared by matrices and
 * vectors of a distributed linear system.
 *
 * Rows are block-distributed with contiguous ownership: rank r owns the global
 * rows [localBegin(), localBegin() + localSize()). A rank may additionally
 * reference rows owned by others ("ghost rows"); ghost slots are appended
 * after the owned block, so a distributed vector is one contiguous array
 * [owned values | ghost values] and every hot kernel (SpMV, Gauss-Seidel
 * sweeps) works with plain local indices and pure local-memory access.
 *
 * The communication plan is derived purely from matrix sparsity (which global
 * columns a rank references), so the module knows nothing about meshes or
 * application-level halo exchangers.
 *
 * Layouts are cheap to copy (shared implementation). Not thread-safe:
 * updateGhosts() uses internal scratch buffers.
 */
class VectorLayout {
public:
    /// Empty layout on MPI_COMM_SELF (serial fallback).
    VectorLayout();

    /// Contiguous ownership of `n_local` of `n_global` rows; the per-rank
    /// start offset comes from a prefix sum (collective: MPI_Exscan).
    VectorLayout(MPI_Comm comm, GlobalIndex n_global, LocalIndex n_local);

    MPI_Comm comm() const;
    int rank() const;
    int nprocs() const;

    GlobalIndex globalSize() const;
    LocalIndex localSize() const;
    GlobalIndex localBegin() const;

    LocalIndex ghostSize() const;
    /// Sorted, unique global ids of the ghost rows.
    const std::vector<GlobalIndex>& ghostGlobalIds() const;

    /// Local slot of a global row (owned block first, ghosts after);
    /// kInvalidLocalIndex when the row is neither owned nor a declared ghost.
    /// Setup-time helper — hot kernels use precomputed local indices.
    LocalIndex localIndex(GlobalIndex global_row) const;

    /// True when both layouts describe the same distribution and ghost set
    /// (pointer-equal fast path, semantic fallback).
    bool compatibleWith(const VectorLayout& other) const;

    /**
     * @brief Declares the ghost rows this rank reads and builds the
     * point-to-point exchange plan (collective on the layout communicator).
     *
     * Called by matrix assembly from the set of foreign columns it references;
     * a layout without ghosts is valid as-is (no call needed).
     * @param sorted_unique_ghost_ids ascending, unique, foreign global rows.
     */
    void setGhosts(const std::vector<GlobalIndex>& sorted_unique_ghost_ids);

    /**
     * @brief Refreshes the ghost entries of `values` — a contiguous array of
     * (localSize() + ghostSize()) * block_size doubles — from their owners.
     *
     * Non-blocking send/recv + wait-all: one synchronous MPI round per call.
     */
    void updateGhosts(double* values, int block_size) const;

private:
    struct Plan;
    std::shared_ptr<Plan> plan_;  // shared: copies of a layout share the plan
};

}  // namespace cfd::linalg

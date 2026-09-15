#pragma once

#include <mpi.h>

#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/linalg/operator.hpp"

namespace cfd::linalg {

/**
 * @class CsrMatrix
 * @brief Distributed matrix in compressed row storage with LOCAL (owned +
 * ghost) column indices in the assembled structure, so the SpMV hot loop is a
 * plain local-memory sweep without any global bookkeeping.
 *
 * Rows are distributed contiguously (see VectorLayout);
 */
class CsrMatrix : public LinearOperator {
public:
    // Collective: contiguous block distribution of `n_global_rows`.
    CsrMatrix(MPI_Comm comm, GlobalIndex n_global_rows, LocalIndex n_local_rows)
        : layout_(comm, n_global_rows, n_local_rows) {}

    // --- LinearOperator ---
    GlobalIndex globalRows() const override { return layout_.globalSize(); }
    LocalIndex localRows() const override { return layout_.localSize(); }
    int blockSize() const override { return 1; }
    const VectorLayout& layout() const override { return layout_; }
    Vector makeVector() const override { return Vector(layout_, 1); }
    void apply(const Vector& x, Vector& y, double alpha = 1.0, double beta = 0.0) const override;
    void setZero() noexcept {
        std::fill(values_.begin(), values_.end(), 0.0);
    }
    
    // --- raw access (preconditioners, inspection) ---
    const std::vector<LocalIndex>& rowPtr() const { return row_ptr_; }
    const std::vector<LocalIndex>& cols() const { return cols_; }
    const std::vector<LocalIndex>& diagIndex() const { return diag_idx_; }
    const std::vector<double>& values() const { return values_; }
    double* valuesData() { return values_.data(); }
    const double* valuesData() const { return values_.data(); }
    bool assembled() const { return assembled_; }

    // --- graph-based assembling (n_total = own + ghosts)---
    void assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                  const std::vector<LocalIndex>& dual_graph_off,
                  const std::vector<LocalIndex>& dual_graph_val);

private:
    VectorLayout layout_;
    std::vector<LocalIndex> row_ptr_, cols_, diag_idx_;
    std::vector<double> values_; 
    bool assembled_ = false;
};

}  // namespace cfd::linalg

#pragma once

#include <mpi.h>

#include <vector>

#include "cfd/linalg/operator.hpp"

namespace cfd::linalg {

/**
 * @class BsrMatrix
 * @brief Distributed block matrix: compressed storage over block rows with
 * dense bs x bs row-major blocks and LOCAL (owned + ghost) block column
 * indices.
 *
 * The natural format for implicit CFD: one block row per cell, one block per
 * (cell, neighbour) coupling, bs = number of conservation variables. Hot
 * kernels are specialized for compile-time block sizes 1..8 (dispatch in the
 * .cpp), with a generic fallback for larger blocks.
 *
 * Assembly mirrors CsrMatrix
 */
class BsrMatrix : public LinearOperator {
public:
    // Collective: contiguous distribution of `n_global_block_rows` block rows.
    BsrMatrix(MPI_Comm comm, GlobalIndex n_global_block_rows, LocalIndex n_local_block_rows,
                     int block_size)
    : layout_(comm, n_global_block_rows, n_local_block_rows), bs_(block_size) {}

    // --- LinearOperator ---
    GlobalIndex globalRows() const override { return layout_.globalSize(); }
    LocalIndex localRows() const override { return layout_.localSize(); }
    int blockSize() const override { return bs_; }
    const VectorLayout& layout() const override { return layout_; }
    Vector makeVector() const override { return Vector(layout_, bs_); }
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
    int bs_ = 1;
    std::vector<LocalIndex> row_ptr_, cols_, diag_idx_;
    std::vector<double> values_;
    bool assembled_ = false;
};

}  // namespace cfd::linalg

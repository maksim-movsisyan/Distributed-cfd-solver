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
 * Assembly mirrors CsrMatrix: addBlock() triplets (duplicates accumulate),
 * collective assemble() (structure + ghost pattern + local block indices),
 * then assembleValues() for fixed-pattern reassembly in time stepping.
 */
class BsrMatrix : public LinearOperator {
public:
    /// Collective: contiguous distribution of `n_global_block_rows` block rows.
    BsrMatrix(MPI_Comm comm, GlobalIndex n_global_block_rows, LocalIndex n_local_block_rows,
              int block_size);

    // --- assembly ---
    void reserve(std::size_t n_blocks_hint);
    /// Copies bs*bs row-major doubles; the block belongs to an owned row.
    void addBlock(GlobalIndex block_row, GlobalIndex block_col, const double* block_values);
    /// Full assembly: structure + ghost pattern + values (collective).
    void assemble();
    /// Values-only reassembly into the existing structure (local, cheap).
    void assembleValues();

    // --- LinearOperator ---
    GlobalIndex globalRows() const override { return layout_.globalSize(); }
    LocalIndex localRows() const override { return layout_.localSize(); }
    int blockSize() const override { return bs_; }
    const VectorLayout& layout() const override { return layout_; }
    Vector makeVector() const override { return Vector(layout_, bs_); }
    void apply(const Vector& x, Vector& y) const override;

    // --- raw access (preconditioners, inspection) ---
    /// Size localRows() + 1, counts BLOCKS (not scalars).
    const std::vector<LocalIndex>& rowPtr() const { return row_ptr_; }
    /// Local block column slot per stored block.
    const std::vector<LocalIndex>& cols() const { return cols_; }
    /// nnzBlocks() * bs * bs doubles, row-major within each block.
    const std::vector<double>& values() const { return values_; }
    double* valuesData() { return values_.data(); }
    /// Diagonal block slot per row or kInvalidLocalIndex when absent.
    const std::vector<LocalIndex>& diagIndex() const { return diag_idx_; }
    bool assembled() const { return assembled_; }

private:
    VectorLayout layout_;
    int bs_ = 1;
    std::vector<LocalIndex> row_ptr_, cols_, diag_idx_;
    std::vector<double> values_;
    std::vector<GlobalIndex> coo_rows_, coo_cols_;
    std::vector<double> coo_vals_;
    bool assembled_ = false;
};

}  // namespace cfd::linalg

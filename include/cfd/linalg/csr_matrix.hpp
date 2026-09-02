#pragma once

#include <mpi.h>

#include <vector>

#include "cfd/linalg/operator.hpp"

namespace cfd::linalg {

/**
 * @class CsrMatrix
 * @brief Distributed matrix in compressed row storage with LOCAL (owned +
 * ghost) column indices in the assembled structure, so the SpMV hot loop is a
 * plain local-memory sweep without any global bookkeeping.
 *
 * Assembly workflow:
 *  1. addValue(row, col, v) — COO triplets, duplicates accumulate;
 *  2. assemble() — collective: sorts/merges the structure, derives the ghost
 *     exchange pattern from the referenced foreign columns and localizes all
 *     column ids;
 *  3. for time stepping with a fixed pattern: addValue() again, then
 *     assembleValues() — binary-search insertion into the existing structure,
 *     no global sort, no communication.
 *
 * Rows are distributed contiguously (see VectorLayout); `row` arguments are
 * global ids and must be owned by this rank, `col` may be any global id.
 */
class CsrMatrix : public LinearOperator {
public:
    /// Collective: contiguous block distribution of `n_global_rows`.
    CsrMatrix(MPI_Comm comm, GlobalIndex n_global_rows, LocalIndex n_local_rows);

    // --- assembly ---
    void reserve(std::size_t nnz_hint);
    /// COO triplet (global row must be owned here); duplicates accumulate.
    void addValue(GlobalIndex row, GlobalIndex col, double value);
    /// Full assembly: structure + ghost pattern + values (collective).
    void assemble();
    /// Values-only reassembly into the existing structure (local, cheap).
    void assembleValues();

    // --- LinearOperator ---
    GlobalIndex globalRows() const override { return layout_.globalSize(); }
    LocalIndex localRows() const override { return layout_.localSize(); }
    int blockSize() const override { return 1; }
    const VectorLayout& layout() const override { return layout_; }
    Vector makeVector() const override { return Vector(layout_, 1); }
    void apply(const Vector& x, Vector& y) const override;

    // --- raw access (preconditioners, inspection) ---
    /// Size localRows() + 1.
    const std::vector<LocalIndex>& rowPtr() const { return row_ptr_; }
    /// Local column slot per nonzero (owned block first, ghosts after).
    const std::vector<LocalIndex>& cols() const { return cols_; }
    const std::vector<double>& values() const { return values_; }
    /// In-place value updates for callers that reassemble by hand.
    double* valuesData() { return values_.data(); }
    /// Diagonal slot per row or kInvalidLocalIndex when structurally absent.
    const std::vector<LocalIndex>& diagIndex() const { return diag_idx_; }
    bool assembled() const { return assembled_; }

private:
    VectorLayout layout_;
    std::vector<LocalIndex> row_ptr_, cols_, diag_idx_;
    std::vector<double> values_;
    std::vector<GlobalIndex> coo_rows_, coo_cols_;
    std::vector<double> coo_vals_;
    bool assembled_ = false;
};

}  // namespace cfd::linalg

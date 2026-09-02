#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "cfd/linalg/vector_layout.hpp"

namespace cfd::linalg {

/**
 * @class Vector
 * @brief Distributed block vector: owned entries followed by ghost entries in
 * one contiguous array.
 *
 * A vector carries a block size (`blockSize()` doubles per row slot — 1 for
 * scalar CSR problems, n_vars for BSR systems) and the ghost exchange pattern
 * of the matrix it was created for (see LinearOperator::makeVector).
 *
 * Element-wise kernels act on the whole local array (owned + ghost) without
 * communication; inner products and norms reduce over OWNED entries only
 * (exactly one MPI_Allreduce per call, batched variants fuse several inner
 * products into a single reduction).
 */
class Vector {
public:
    Vector() = default;

    /// Zero-initialized vector over `layout` with `block_size` doubles per row.
    Vector(const VectorLayout& layout, int block_size);

    const VectorLayout& layout() const { return layout_; }
    int blockSize() const { return block_size_; }

    LocalIndex localRows() const { return layout_.localSize(); }
    /// Owned scalar entries (block rows * block size).
    std::size_t ownedScalarCount() const {
        return static_cast<std::size_t>(layout_.localSize()) *
               static_cast<std::size_t>(block_size_);
    }
    /// Owned + ghost scalar entries (full local array).
    std::size_t scalarSize() const { return values_.size(); }

    double* data() { return values_.data(); }
    const double* data() const { return values_.data(); }

    std::span<double> owned() { return {values_.data(), ownedScalarCount()}; }
    std::span<const double> owned() const { return {values_.data(), ownedScalarCount()}; }
    std::span<double> ghosts() { return {values_.data() + ownedScalarCount(), ghostScalarCount()}; }
    std::span<const double> ghosts() const {
        return {values_.data() + ownedScalarCount(), ghostScalarCount()};
    }

    // --- halo ----------------------------------------------------------------
    /// Ghost slots := current values at their owners (one MPI round).
    /// Logically const: the ghost tail is a communication cache.
    void updateGhosts() const {
        layout_.updateGhosts(const_cast<double*>(values_.data()), block_size_);
    }

    // --- element-wise (local, no communication; whole owned+ghost array) -----
    void setZero();
    /// Full deep copy (owned + ghost slots); layouts must be compatible.
    void copyFrom(const Vector& other);
    void scale(double alpha);
    /// *this += alpha * x.
    void axpy(double alpha, const Vector& x);

    // --- reductions (owned entries + one MPI_Allreduce) ----------------------
    /// Euclidean inner product over all ranks; layouts must be compatible.
    double dot(const Vector& other) const;
    double norm2() const;

    /**
     * @brief Several inner products with a single MPI_Allreduce (reduction
     * fusion for Krylov methods): out[i] = a_i . b_i.
     */
    static void batchedDots(std::span<const std::pair<const Vector*, const Vector*>> products,
                            std::span<double> out);

private:
    std::size_t ghostScalarCount() const {
        return static_cast<std::size_t>(layout_.ghostSize()) *
               static_cast<std::size_t>(block_size_);
    }

    VectorLayout layout_;
    int block_size_ = 0;
    std::vector<double> values_;
};

}  // namespace cfd::linalg

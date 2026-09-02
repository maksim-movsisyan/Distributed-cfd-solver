#pragma once

#include "cfd/linalg/vector.hpp"

namespace cfd::linalg {

/**
 * @class LinearOperator
 * @brief Abstract distributed operator y = A x acting on block vectors.
 *
 * The matrix-free seam of the module: Krylov solvers and generic code see only
 * this interface, so anything that can multiply — explicit CSR/BSR matrices,
 * field-split composites, Jacobian-free Newton–Krylov approximations — plugs
 * in without touching the solvers. One virtual call per application, zero
 * virtual calls inside the kernel.
 *
 * Operators are assumed square (rows == columns), as produced by implicit
 * discretizations.
 */
class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    virtual GlobalIndex globalRows() const = 0;
    virtual LocalIndex localRows() const = 0;
    /// Doubles per row slot: 1 for CSR, the block size for BSR.
    virtual int blockSize() const = 0;
    virtual const VectorLayout& layout() const = 0;

    /// New zero vector compatible with this operator (shares the ghost plan).
    virtual Vector makeVector() const = 0;

    /**
     * @brief y = A x (owned part of y; ghost slots of y are not touched).
     *
     * Contract:
     *  - the ghost slots of `x` must be CURRENT (caller calls
     *    x.updateGhosts() first) — kernels index straight into them;
     *  - `x` and `y` must be distinct vectors (no aliasing).
     */
    virtual void apply(const Vector& x, Vector& y) const = 0;
};

}  // namespace cfd::linalg

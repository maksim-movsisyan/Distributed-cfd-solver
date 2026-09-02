#pragma once

#include "cfd/linalg/operator.hpp"
#include "cfd/linalg/vector.hpp"

namespace cfd::linalg {

/**
 * @class Preconditioner
 * @brief Abstract M ≈ A applied as z = M^{-1} r.
 *
 * setup() binds the preconditioner to a concrete operator, typically right
 * after matrix assembly (and again whenever the matrix values change enough
 * to warrant refactoring). apply() is const so solvers can cache and reuse
 * configured preconditioners.
 *
 * Contract for apply(r, z):
 *  - the ghost slots of `r` are never read;
 *  - on exit the ghost slots of `z` are CURRENT (a halo exchange at the end
 *    is the natural implementation), so operators can consume z directly.
 */
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const LinearOperator& op) = 0;
    virtual void apply(const Vector& r, Vector& z) const = 0;
};

}  // namespace cfd::linalg

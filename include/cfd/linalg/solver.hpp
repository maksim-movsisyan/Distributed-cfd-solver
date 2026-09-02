#pragma once

#include "cfd/linalg/operator.hpp"
#include "cfd/linalg/preconditioner.hpp"
#include "cfd/linalg/types.hpp"
#include "cfd/linalg/vector.hpp"

namespace cfd::linalg {

/**
 * @class IterativeSolver
 * @brief Abstract iterative solver for A x = b over the LinearOperator /
 * Preconditioner seams.
 *
 * Solvers are right-preconditioned by convention: the preconditioner acts on
 * residuals, the iterate stays in the true variable (no unpreconditioning of
 * the solution, and the convergence test runs on the true residual).
 *
 * Concrete solvers hold reusable workspace; one instance per system is the
 * intended (single-threaded per rank) usage pattern.
 */
class IterativeSolver {
public:
    virtual ~IterativeSolver() = default;

    virtual const char* name() const = 0;

    /**
     * @brief Solves A x = b starting from the initial guess in `x`
     * (overwritten with the solution).
     *
     * The ghost slots of `x` and `b` may be stale; they are refreshed before
     * use. `M` must have been set up for `A`.
     */
    virtual IterationResult solve(const LinearOperator& A, const Preconditioner& M, Vector& x,
                                  const Vector& b) = 0;

    const SolverParams& params() const { return params_; }
    SolverParams& params() { return params_; }

    const IterationResult& lastResult() const { return last_; }

protected:
    SolverParams params_;
    IterationResult last_;
};

}  // namespace cfd::linalg

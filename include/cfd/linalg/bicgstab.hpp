#pragma once

#include "cfd/linalg/solver.hpp"

namespace cfd::linalg {

/**
 * @class BiCGSTAB
 * @brief Right-preconditioned stabilized bi-conjugate gradient method
 * (van der Vorst) for general (non-symmetric) distributed systems.
 *
 * Reduction pattern: two MPI_Allreduce rounds per iteration. The residual
 * norm and the next rho are reconstructed from a single 5-way fused inner
 * product, and the monitored norm is periodically re-evaluated from the
 * actual recurrence vector to keep the stopping test honest.
 */
class BiCGSTAB : public IterativeSolver {
public:
    const char* name() const override { return "BiCGSTAB"; }

    IterationResult solve(const LinearOperator& A, const Preconditioner& M, Vector& x,
                          const Vector& b) override;

private:
    void ensureWorkspace(const Vector& prototype);

    Vector r_, r0_, p_, v_, s_, t_, phat_, shat_;
};

}  // namespace cfd::linalg

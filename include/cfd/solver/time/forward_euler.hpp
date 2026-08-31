// Forward Euler (first-order explicit) time integration policy.
//
//   u^{n+1} = u^n - alpha * R(u^n),   alpha = cfl / lambda
#pragma once

#include "cfd/solver/fields/update_ops.hpp"

namespace cfd::solver::time {

/**
 * @class ForwardEuler
 * @brief First-order explicit time integrator over all update blocks.
 * @tparam Op Residual operator (the Solver instantiation).
 */
template <typename Op>
class ForwardEuler {
public:
    using Operator = Op;

    static constexpr bool kNeedsPrevSnapshot = false;
    static constexpr const char* name() noexcept { return "FORWARD_EULER"; }

    void advance(Op& op) const noexcept {
        op.evaluate_residual(op.u_slots());
        op.compute_dt();

        fields::block_sub_axpy(op.stage_slots(), op.u_slots(), op.res_slots(),
                               op.alpha(), op.n_owned());
        op.post_stage(op.stage_slots());
        op.ping_pong();
    }
};

} // namespace cfd::solver::time

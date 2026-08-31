// SSP-RK3 (Shu–Osher three-stage, TVD) explicit time integration policy.
//
//   Stage 1: u(1)      = u^n - alpha * R(u^n)
//   Stage 2: u(2)      = 3/4 u^n + 1/4 (u(1) - alpha * R(u(1)))
//   Stage 3: u^{n+1}   = 1/3 u^n + 2/3 (u(2) - alpha * R(u(2))
#pragma once

#include "cfd/solver/fields/update_ops.hpp"

namespace cfd::solver::time {

/**
 * @class SspRk3
 * @brief Third-order strong-stability-preserving Runge–Kutta integrator over all update blocks.
 * @tparam Op Residual operator (the Solver instantiation).
 */
template <typename Op>
class SspRk3 {
public:
    using Operator = Op;

    static constexpr bool kNeedsPrevSnapshot = true; // holds u^n across stages
    static constexpr const char* name() noexcept { return "SSP_RK3"; }

    void advance(Op& op) const noexcept {
        // Stage 1: u(1) = u^n - alpha * R(u^n)
        op.evaluate_residual(op.u_slots());
        op.compute_dt();
        fields::block_copy(op.prev_slots(), op.u_slots(), op.n_owned());
        fields::block_sub_axpy(op.stage_slots(), op.u_slots(), op.res_slots(),
                               op.alpha(), op.n_owned());
        op.post_stage(op.stage_slots());

        // Stage 2: u(2) = 3/4 u^n + 1/4 (u(1) - alpha * R(u(1)))
        op.evaluate_residual(op.stage_slots());
        fields::block_ssp_combine(op.u_slots(), 0.75, op.prev_slots(), 0.25,
                                  op.stage_slots(), op.res_slots(),
                                  op.alpha(), op.n_owned());
        op.post_stage(op.u_slots());

        // Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u(2) - alpha * R(u(2)))
        op.evaluate_residual(op.u_slots());
        fields::block_ssp_combine(op.stage_slots(), 1.0 / 3.0, op.prev_slots(), 2.0 / 3.0,
                                  op.u_slots(), op.res_slots(),
                                  op.alpha(), op.n_owned());
        op.post_stage(op.stage_slots());

        op.ping_pong();
    }
};

} // namespace cfd::solver::time

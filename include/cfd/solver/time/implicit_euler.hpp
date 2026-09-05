// Backward Euler (first-order implicit) time integration policy.
//
//   (V_i/dt_i + dR/dU) du = -R(u^n),    u^{n+1} = u^n + du
//
// The linear system is assembled natively from the mesh (structure built once
// by implicit::MeanFlowSystem) and solved by distributed BiCGSTAB with a
// hybrid SGS preconditioner. The matrix contains the MEAN FLOW only; physics
// modules (turbulence transport) are frozen during the implicit step in this
// first version — their slots are copied unchanged (see the notice below).
//
// STATUS (work in progress): the assembly pipeline is in place and its
// interior Jacobian is finite-difference-verified against the residual; the
// linear solve on strongly anisotropic supersonic cases does not yet reach
// the configured tolerance within a practical iteration budget — see the
// notes in implicit_system.hpp (diagonal-dominance modification) before
// production use.
#pragma once

#include <memory>

#include "cfd/mpi/log.hpp"
#include "cfd/solver/implicit/implicit_system.hpp"

namespace cfd::solver::time {

/**
 * @class BackwardEuler
 * @brief Implicit Euler integrator for the mean flow over the linalg module.
 * @tparam Op Residual operator (the Solver instantiation).
 */
template <typename Op>
class BackwardEuler {
public:
    using Operator = Op;

    static constexpr bool kNeedsPrevSnapshot = false;
    static constexpr const char* name() noexcept { return "BACKWARD_EULER"; }

    void advance(Op& op) noexcept {
        // 1. Residual and state services at u^n (primitives, halos, BC ghosts —
        //    the Jacobian sweep gathers these directly).
        op.evaluate_residual(op.u_slots());
        op.compute_dt();

        // 2. One-time system structure from the mesh adjacency.
        if (!system_) {
            system_ = std::make_unique<implicit::MeanFlowSystem>(
                op.mesh(), op.mpi_comm(), op.config().implicit_tolerance,
                op.config().implicit_max_iterations);
            system_->build();
            if constexpr (Op::kHasModules) {
                mpi::log_stat("BACKWARD_EULER: physics modules are frozen during the "
                              "implicit step (mean-flow matrix only)");
            }
        }

        // 3. Assemble (V/dt + dR/dU) natively: in-place block fills.
        system_->begin_assembly();
        op.assemble_mean_flow_jacobian(*system_);
        const auto n_own = op.n_owned();
        const double* CFD_RESTRICT dt = op.local_dt();
        const double* CFD_RESTRICT vol = op.mesh().cell_volume.data();
        for (std::size_t c = 0; c < n_own; ++c) {
            system_->add_time_term(static_cast<LocalIndex>(c), vol[c] / dt[c]);
        }

        // 4. Solve for the increment du.
        system_->set_rhs_from_residual(op.res_slots());
        system_->solve();
        if (!system_->linear_converged()) {
            mpi::log_stat("BACKWARD_EULER: linear solve stopped at %d iterations "
                          "(using the current increment)",
                          system_->linear_iterations());
        } else if (g_verbose >= 1) {
            mpi::log_stat("BACKWARD_EULER: linear solve %d iterations",
                          system_->linear_iterations());
        }

        // 5. Update: mean flow u^{n+1} = u^n + du; module slots copied frozen.
        const double* CFD_RESTRICT du = system_->increment();
        const auto u = op.u_slots();
        const auto stage = op.stage_slots();
        for (std::size_t v = 0; v < 5; ++v) {
            double* CFD_RESTRICT dst = stage[v];
            const double* CFD_RESTRICT src = u[v];
            for (std::size_t c = 0; c < n_own; ++c) {
                dst[c] = src[c] + du[c * implicit::MeanFlowSystem::kNumVars + v];
            }
        }
        if constexpr (Op::kHasModules) {
            for (std::size_t v = 5; v < u.size(); ++v) {
                double* CFD_RESTRICT dst = stage[v];
                const double* CFD_RESTRICT src = u[v];
                for (std::size_t c = 0; c < n_own; ++c) {
                    dst[c] = src[c];
                }
            }
        }

        op.post_stage(op.stage_slots());
        op.ping_pong();
    }

private:
    std::unique_ptr<implicit::MeanFlowSystem> system_;
};

}  // namespace cfd::solver::time

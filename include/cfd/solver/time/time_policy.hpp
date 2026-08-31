// Time integration POLICY contract.
//
// A time policy advances the complete state (all registered update blocks)
// by one step. It is a compile-time policy of the Solver — like FluxPolicy and
// ReconPolicy — so stage algebra inlines fully; unlike them it is a stateful
// instance (owned scratch / future implicit data) operating on the solver as
// its residual operator.
//
// The operator interface required from the Solver (all public):
//   op.evaluate_residual(state_slots)   — R(state), full pipeline
//   op.compute_dt()                     — fills dt[]/alpha[]
//   op.u_slots()/stage_slots()/prev_slots()/res_slots() — update block pointers
//   op.alpha(), op.n_owned(), op.ping_pong()
//   op.post_stage(state_slots)          — module positivity clamps on the
//                                         buffer a stage just wrote
#pragma once

#include <concepts>

namespace cfd::solver::time {

/**
 * @concept TimeIntegrationPolicy
 * @brief Static interface contract for time integration policies.
 * @tparam T Policy type (e.g. ForwardEuler<Op>, SspRk3<Op>).
 */
template <typename T>
concept TimeIntegrationPolicy = requires(typename T::Operator& op, T t) {
    // The residual operator this policy advances (the Solver instantiation).
    typename T::Operator;

    // 1. Static compile-time metadata
    { T::kNeedsPrevSnapshot } -> std::convertible_to<bool>;
    { T::name() } -> std::convertible_to<const char*>;

    // 2. One full step: stages + block updates + ping-pong
    { t.advance(op) } noexcept;
};

} // namespace cfd::solver::time

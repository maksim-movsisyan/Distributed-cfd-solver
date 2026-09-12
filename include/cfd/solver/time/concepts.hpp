// Time integration POLICY contract.
#pragma once

#include <concepts>
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"

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
    { T::kNeedsMatrix } -> std::convertible_to<bool>;
    { T::kAuxConnectivity } -> std::convertible_to<mesh::AuxConnType>;
    { T::kAuxGeometry } -> std::convertible_to<mesh::AuxGeomType>;
    { T::name() } -> std::convertible_to<const char*>;

    // 2. One full step: stages + block updates + ping-pong
    { t.advance(op) } noexcept;
};

} // namespace cfd::solver::time

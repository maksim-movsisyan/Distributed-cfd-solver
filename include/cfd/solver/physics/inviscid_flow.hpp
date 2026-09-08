#pragma once

#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"

namespace cfd::solver::physics {

/**
 * @struct InviscidFlow
 * @brief Euler equations: pure inviscid mean flow.
 */
struct InviscidFlow {
    static constexpr std::size_t kNumVars = static_cast<std::size_t>(constants::kNumVars);
    static constexpr bool kHasViscous = false;
    static constexpr bool kNeedsGradients = false;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "INVISCID_FLOW"; }

    // not used for mean-flow
    static constexpr std::size_t kNumExtraVars = 0;
    static constexpr bool kHasEddyViscosity = false;
    static constexpr bool kNeedsFaceMdot = false;
    static constexpr bool kNeedsWallDist = false;
};

static_assert(PhysicsGeneral<InviscidFlow>);

} // namespace cfd::solver::physics

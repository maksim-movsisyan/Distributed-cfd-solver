#pragma once

#include <cmath>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"

namespace cfd::solver::physics {

/**
 * @struct ViscousFlow
 * @brief Navier-Stokes equations: pure viscous mean flow.
 */
struct ViscousFlow {
    double prandtl = constants::kAirPrandtl; ///< Molecular Prandtl number [-]
    
    static constexpr std::size_t kNumVars = static_cast<std::size_t>(constants::kNumVars);
    static constexpr bool kHasViscous = true;
    static constexpr bool kNeedsGradients = true;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::FaceCellDistanceInv|
                                                      mesh::AuxGeomType::FaceCellDistanceVector; 
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "VISCOUS_FLOW"; }

    // not used for mean-flow
    static constexpr std::size_t kNumExtraVars = 0;
    static constexpr bool kHasEddyViscosity = false;
    static constexpr bool kNeedsFaceMdot = false;
    static constexpr bool kNeedsWallDist = false;
};

static_assert(PhysicsGeneral<ViscousFlow>);

} // namespace cfd::solver::physics
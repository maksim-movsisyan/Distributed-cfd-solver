#pragma once

#include <concepts>
#include <cstddef>

#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"

namespace cfd::solver::physics {

/**
 * @concept PhysicsGeneral
 * @brief Static interface contract for general physics - mean flow + modules.
 */
template <typename P>
concept PhysicsGeneral = requires() {
    // 1. Equation-set metadata
    { P::kNumVars } -> std::convertible_to<std::size_t>;  // mean-flow variables (or additional variables in modules)
    { P::kHasViscous } -> std::convertible_to<bool>;      // adds viscous face fluxes
    { P::kNeedsGradients } -> std::convertible_to<bool>;  // forces cell gradients
    { P::kAuxGeometry } -> std::convertible_to<mesh::AuxGeomType>;
    { P::kAuxConnectivity } -> std::convertible_to<mesh::AuxConnType>;
    { P::name() } -> std::convertible_to<const char*>;

    // 2. Physics-stack composition metadata (zero for plain equation sets)
    { P::kNumExtraVars } -> std::convertible_to<std::size_t>; // module variables (for generalization stack, 
                                                              // use kNumVars in concrete implementations)
    { P::kHasEddyViscosity } -> std::convertible_to<bool>;    // provides mut data
    { P::kNeedsFaceMdot } -> std::convertible_to<bool>;       // consumes face mass flux
    { P::kNeedsWallDist } -> std::convertible_to<bool>;       // consumes wall distance

};

} //namespace::cfd::solver::physics
#pragma once

#include <concepts>
#include <cstddef>

#include "cfd/mesh/localmesh.hpp"

namespace cfd::solver::physics {

/**
 * @concept PhysicsGeneral
 * @brief Static interface contract for general physics - mean flow + modules.
 */
template <typename P>
concept PhysicsGeneral = requires(const mesh::MeshPart& mp) {
    // 1. Equation-set metadata
    { P::kNumVars } -> std::convertible_to<std::size_t>;  // mean-flow variables
    { P::kHasViscous } -> std::convertible_to<bool>;      // adds viscous face fluxes
    { P::kNeedsGradients } -> std::convertible_to<bool>;  // forces cell gradients
    { P::name() } -> std::convertible_to<const char*>;

    // 2. Physics-stack composition metadata (zero for plain equation sets)
    { P::kNumExtraVars } -> std::convertible_to<std::size_t>; // module variables (for generalization stack, 
                                                              // use kNumVars in concrete implementations)
    { P::kHasEddyViscosity } -> std::convertible_to<bool>;    // provides mut data
    { P::kNeedsFaceMdot } -> std::convertible_to<bool>;       // consumes face mass flux
    { P::kNeedsWallDist } -> std::convertible_to<bool>;       // consumes wall distance

    // 3. Mesh-fixed geometry type & builder
    typename P::Geometry;
    { P::build_geometry(mp) } -> std::same_as<typename P::Geometry>;

};

} //namespace::cfd::solver::physics
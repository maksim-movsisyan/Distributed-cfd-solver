// Reconstruction POLICY contract and shared context types.
//
// A reconstruction policy maps cell-centred primitives (+ gradients and
// limiters when needed) to high-order primitive states on both sides of a face:
//   qL = q_c0 + phi_c0 * (grad(q_c0) . r_0f)
//   qR = q_c1 + phi_c1 * (grad(q_c1) . r_1f)
#pragma once

#include <concepts>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::recon {

/**
 * @struct ReconField
 * @brief Context containing cell primitive values, gradients, and limiters.
 */
struct ReconField {
    fields::ConstPrimitiveView q;        ///< Cell primitive states
    fields::ConstPrimitiveGradView grad; ///< Cell gradients (valid if kNeedsGradients == true)
    fields::ConstPrimitiveView phi;      ///< Cell gradient limiters in [0, 1]
};

/**
 * @concept ReconstructionPolicy
 * @brief Static interface contract for spatial reconstruction policies.
 */
template <typename R>
concept ReconstructionPolicy = requires(
    const R& r,
    const ReconField& s,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxGeometry& aux_geom,
    std::size_t f,
    std::size_t c0,
    std::size_t c1,
    std::size_t cg,
    double qL[constants::kNumVars],
    double qR[constants::kNumVars]) {

    // 1. Static compile-time metadata
    { R::kNeedsGradients } -> std::convertible_to<bool>;
    { R::kAuxGeometry } -> std::convertible_to<mesh::AuxGeomType>;
    { R::kAuxConnectivity } -> std::convertible_to<mesh::AuxConnType>;
    { R::name() } -> std::convertible_to<const char*>;

    // 2. Evaluation on interior faces [0, n_inner_faces)
    { R::face_states(s, mesh, aux_geom, f, c0, c1, qL, qR) } noexcept;

    // 3. Evaluation on boundary faces [n_inner_faces, n_faces)
    { R::boundary_face_states(s, mesh, aux_geom, f, c0, cg, qL, qR) } noexcept;
};

} // namespace cfd::solver::recon
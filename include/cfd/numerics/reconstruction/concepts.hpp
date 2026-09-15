#pragma once

#include <concepts>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"

namespace cfd::numerics::recon {

/**
 * @struct ReconBatchField
 * @brief General reconstruction context fo NVars variables. Keep SoA pointer to reconstruction fields
 */
template <std::size_t NVars>
struct ReconBatchField {
    const double* CFD_RESTRICT q[NVars]{};
    const double* CFD_RESTRICT grad_x[NVars]{};
    const double* CFD_RESTRICT grad_y[NVars]{};
    const double* CFD_RESTRICT grad_z[NVars]{};
    const double* CFD_RESTRICT phi[NVars]{};

    constexpr ReconBatchField() noexcept = default;

    constexpr ReconBatchField(
        const double* CFD_RESTRICT const* q_in,
        const double* CFD_RESTRICT const* gx_in = nullptr,
        const double* CFD_RESTRICT const* gy_in = nullptr,
        const double* CFD_RESTRICT const* gz_in = nullptr,
        const double* CFD_RESTRICT const* phi_in = nullptr) noexcept {

            for (std::size_t v = 0; v < NVars; ++v) {
            if (q_in)   q[v] = q_in[v];
            if (gx_in)  grad_x[v] = gx_in[v];
            if (gy_in)  grad_y[v] = gy_in[v];
            if (gz_in)  grad_z[v] = gz_in[v];
            if (phi_in) phi[v] = phi_in[v];
        }
    }
};

/**
 * @concept ReconstructionPolicy
 * @brief General concept with multivariables support
 */
template <typename R, std::size_t NVars>
concept ReconstructionPolicy = requires(
    const ReconBatchField<NVars>& s,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxGeometry& aux_geom,
    const mesh::MeshAuxConnectivity& aux_conn,
    std::size_t f,
    std::size_t c0,
    std::size_t c1,
    std::size_t cg,
    double* CFD_RESTRICT qL,
    double* CFD_RESTRICT qR,
    double* const* CFD_RESTRICT phi_out) {

    { R::kNeedsGradients } -> std::convertible_to<bool>;
    { R::kAuxGeometry } -> std::convertible_to<mesh::AuxGeomType>;
    { R::kAuxConnectivity } -> std::convertible_to<mesh::AuxConnType>;
    { R::name() } -> std::convertible_to<const char*>;
    { R::limiter_name() } -> std::convertible_to<const char*>;

    { R::template face_states<NVars>(s, mesh, aux_geom, f, c0, c1, qL, qR) } noexcept;

    { R::template boundary_face_states<NVars>(s, mesh, aux_geom, f, c0, cg, qL, qR) } noexcept;
};

} // namespace cfd::numerics::recon
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"

namespace cfd::solver::incompressible {

/**
 * @struct RhieChowKernel
 * @brief Lightweight functor for evaluating face mass flux via Rhie-Chow interpolation.
 * Stores raw restricted pointers to eliminate pointer-chasing and function call overhead.
 */
struct RhieChowKernel {
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::FaceCellDistanceInv|
                                                      mesh::AuxGeomType::FaceCellDistanceVector|
                                                      mesh::AuxGeomType::FaceInterpolationWeights; 

    const double* CFD_RESTRICT u{nullptr};
    const double* CFD_RESTRICT v{nullptr};
    const double* CFD_RESTRICT w{nullptr};
    const double* CFD_RESTRICT p{nullptr};
    const double* CFD_RESTRICT gpx{nullptr};
    const double* CFD_RESTRICT gpy{nullptr};
    const double* CFD_RESTRICT gpz{nullptr};
    const double* CFD_RESTRICT d_coeff{nullptr}; // d = V_cell / a_P
    double rho{1.0};
    const mesh::MeshPart* mesh{nullptr};

    const mesh::MeshAuxGeometry* aux_geom{nullptr};
    const double* CFD_RESTRICT face_cell_dist_x{nullptr};
    const double* CFD_RESTRICT face_cell_dist_y{nullptr};
    const double* CFD_RESTRICT face_cell_dist_z{nullptr};
    const double* CFD_RESTRICT inv_d{nullptr};
    const double* CFD_RESTRICT face_weight{nullptr};

    /**
     * @brief Computes mass flux [kg/s] through the interior face `f`.
     * Face normal is assumed to point from owner to neighbor.
     */
    [[nodiscard]] inline double operator()(const std::size_t f) const noexcept {
        const auto& m = *mesh;

        const auto owner = static_cast<std::size_t>(m.face_owner[f]);
        const auto neigh = static_cast<std::size_t>(m.face_neigh[f]);

        // Cell centroid distance vector: P (owner) -> N (neighbor)
        const double dx = face_cell_dist_x[f];
        const double dy = face_cell_dist_y[f];
        const double dz = face_cell_dist_z[f];
        const double inv_dist = inv_d[f];

        // Unit vector connecting cell centroids
        const double ex = dx * inv_dist;
        const double ey = dy * inv_dist;
        const double ez = dz * inv_dist;

        // Unit face normal and area
        const double nx = m.face_normal_x[f];
        const double ny = m.face_normal_y[f];
        const double nz = m.face_normal_z[f];
        const double area = m.face_area[f];

        // Geometric distance-based weighting factor (fallback to 0.5)
        const double w_own = face_weight[f];
        const double w_ngh = 1.0 - w_own;

        // 1. Linearly interpolated convective velocity
        const double u_bar = w_own * u[owner] + w_ngh * u[neigh];
        const double v_bar = w_own * v[owner] + w_ngh * v[neigh];
        const double w_bar = w_own * w[owner] + w_ngh * w[neigh];
        const double vn_geom = u_bar * nx + v_bar * ny + w_bar * nz;

        // 2. Averaged cell-centered pressure gradient at the face
        const double gpx_bar = w_own * gpx[owner] + w_ngh * gpx[neigh];
        const double gpy_bar = w_own * gpy[owner] + w_ngh * gpy[neigh];
        const double gpz_bar = w_own * gpz[owner] + w_ngh * gpz[neigh];

        // Projected gradient along the centroid-to-centroid direction
        const double grad_p_bar_proj = gpx_bar * ex + gpy_bar * ey + gpz_bar * ez;

        // 3. Compact finite-difference gradient across the face
        const double grad_p_compact = (p[neigh] - p[owner]) * inv_dist;

        // 4. Interpolated momentum coefficient: d = V_cell / a_P
        const double d_f = w_own * d_coeff[owner] + w_ngh * d_coeff[neigh];

        // Non-orthogonality angle: cos(theta) = e_PN . n_f
        const double cos_theta = ex * nx + ey * ny + ez * nz;
        const double non_ortho_factor = (cos_theta > 0.05) ? cos_theta : 0.05;

        // Rhie-Chow normal velocity correction
        const double vn_corr = -d_f * (grad_p_compact - grad_p_bar_proj) * non_ortho_factor;

        // Total mass flux = rho * (Vn_geom + Vn_corr) * Area
        return rho * (vn_geom + vn_corr) * area;
    }
};

/**
 * @brief Bulk kernel: calculates Rhie-Chow mass fluxes for all interior faces.
 * Uses raw RESTRICT pointers to allow vectorization and eliminate cache/stack overhead.
 * 
 * @param[out] m_dot   Face mass fluxes array [0, n_inner_faces).
 */
inline void compute_rhie_chow_fluxes(
    double* CFD_RESTRICT m_dot,
    const double* CFD_RESTRICT u,
    const double* CFD_RESTRICT v,
    const double* CFD_RESTRICT w,
    const double* CFD_RESTRICT p,
    const double* CFD_RESTRICT gpx,
    const double* CFD_RESTRICT gpy,
    const double* CFD_RESTRICT gpz,
    const double* CFD_RESTRICT d_coeff,
    const double rho,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxGeometry& aux_geom) noexcept {

    const RhieChowKernel flux_kernel{
        .u       = u,
        .v       = v,
        .w       = w,
        .p       = p,
        .gpx     = gpx,
        .gpy     = gpy,
        .gpz     = gpz,
        .d_coeff = d_coeff,
        .rho     = rho,
        .mesh    = &mesh,
        .aux_geom = &aux_geom,
        .face_cell_dist_x = aux_geom.face_cell_dist_x.data(),
        .face_cell_dist_y = aux_geom.face_cell_dist_y.data(),
        .face_cell_dist_z = aux_geom.face_cell_dist_z.data(),
        .inv_d = aux_geom.face_cell_dist_inv.data(),
        .face_weight = aux_geom.face_interp_weight.data()
    };

    const std::size_t n_inner = static_cast<std::size_t>(mesh.n_inner_faces);

    for (std::size_t f = 0; f < n_inner; ++f) {
        m_dot[f] = flux_kernel(f);
    }
}

} // namespace cfd::solver::incompressible
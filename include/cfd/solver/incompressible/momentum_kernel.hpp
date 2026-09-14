#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"

namespace cfd::solver::incompressible {

struct MomentumFieldViews {
    const double* CFD_RESTRICT u{nullptr};
    const double* CFD_RESTRICT v{nullptr};
    const double* CFD_RESTRICT w{nullptr};

    // Pressure gradient fields
    const double* CFD_RESTRICT gpx{nullptr};
    const double* CFD_RESTRICT gpy{nullptr};
    const double* CFD_RESTRICT gpz{nullptr};

    // Velocity gradients
    const double* CFD_RESTRICT gu_x{nullptr};
    const double* CFD_RESTRICT gu_y{nullptr};
    const double* CFD_RESTRICT gu_z{nullptr};

    const double* CFD_RESTRICT gv_x{nullptr};
    const double* CFD_RESTRICT gv_y{nullptr};
    const double* CFD_RESTRICT gv_z{nullptr};

    const double* CFD_RESTRICT gw_x{nullptr};
    const double* CFD_RESTRICT gw_y{nullptr};
    const double* CFD_RESTRICT gw_z{nullptr};

    // Limiters
    const double* CFD_RESTRICT lim_u{nullptr};
    const double* CFD_RESTRICT lim_v{nullptr};
    const double* CFD_RESTRICT lim_w{nullptr};

    // Previous time-step velocities: t^n (BDF1 / BDF2)
    const double* CFD_RESTRICT u_old{nullptr};
    const double* CFD_RESTRICT v_old{nullptr};
    const double* CFD_RESTRICT w_old{nullptr};

    // Two time-steps back velocities: t^{n-1} (only for BDF2)
    const double* CFD_RESTRICT u_older{nullptr};
    const double* CFD_RESTRICT v_older{nullptr};
    const double* CFD_RESTRICT w_older{nullptr};
};

/**
 * @struct MomentumMatrixViews
 * @brief Output views for the assembled momentum SLAE.
 * Matrix operator A is stored in LDU format (upper, lower, diag) and shared across u, v, w.
 */
struct MomentumMatrixViews {
    double* CFD_RESTRICT diag{nullptr};   ///< Main diagonal a_P [n_own]
    double* CFD_RESTRICT upper{nullptr};  ///< a_PN coeff for neighbor in owner eq [n_inner_faces]
    double* CFD_RESTRICT lower{nullptr};  ///< a_NP coeff for owner in neighbor eq [n_inner_faces]
    double* CFD_RESTRICT rhs_u{nullptr};  ///< Right-hand side for u-momentum [n_own]
    double* CFD_RESTRICT rhs_v{nullptr};  ///< Right-hand side for v-momentum [n_own]
    double* CFD_RESTRICT rhs_w{nullptr};  ///< Right-hand side for w-momentum [n_own]
    double* CFD_RESTRICT d_coeff{nullptr};///< Out: Rhie-Chow momentum coefficient d = V / a_P [n_own]
};

/**
 * @class MomentumKernel
 * @brief Parameterized momentum operator assembler for collocated unstructured FVM.
 * 
 * @tparam TimeMode    SteadyMode or UnsteadyMode policy.
 * @tparam ReconPolicy Reconstruction policy (e.g. Muscl<LimiterPolicy>).
 */
template <typename TimeMode, typename ReconPolicy>
class MomentumKernel {
public:
    static constexpr bool kIsUnsteady = requires {
        requires TimeMode::kIsUnsteady == true;
    };

    /**
     * @brief Phase 1: Assembles interior face fluxes (convection & diffusion).
     * 
     * @param[in]  fields   Velocity, gradients, limiters views.
     * @param[in]  m_dot    Interior face mass fluxes [0, n_inner_faces).
     * @param[out] mat      Momentum matrix and RHS views.
     * @param[in]  mu       Dynamic viscosity (Pa*s).
     * @param[in]  mesh     Local partitioned mesh.
     * @param[in]  aux_geom Precomputed auxiliary geometric metrics.
     */
    static void assemble_interior_faces(
        const MomentumFieldViews& fields,
        const double* CFD_RESTRICT m_dot,
        MomentumMatrixViews& mat,
        const double mu,
        const mesh::MeshPart& mesh,
        const mesh::MeshAuxGeometry& aux_geom) noexcept {

        const std::size_t n_inner = static_cast<std::size_t>(mesh.n_inner_faces);

        const double* CFD_RESTRICT dist_x = aux_geom.face_cell_dist_x.data();
        const double* CFD_RESTRICT dist_y = aux_geom.face_cell_dist_y.data();
        const double* CFD_RESTRICT dist_z = aux_geom.face_cell_dist_z.data();
        const double* CFD_RESTRICT inv_d  = aux_geom.face_cell_dist_inv.data();
        const double* CFD_RESTRICT weight = aux_geom.face_interp_weight.data();

        // Prepare batch for high-order face states reconstruction (3 components: u, v, w)
        const double* q_ptrs[3]   = { fields.u, fields.v, fields.w };
        const double* gx_ptrs[3]  = { fields.gu_x, fields.gv_x, fields.gw_x };
        const double* gy_ptrs[3]  = { fields.gu_y, fields.gv_y, fields.gw_y };
        const double* gz_ptrs[3]  = { fields.gu_z, fields.gv_z, fields.gw_z };
        const double* lim_ptrs[3] = { fields.lim_u, fields.lim_v, fields.lim_w };

        const numerics::recon::ReconBatchField<3> batch{
            .q      = q_ptrs,
            .grad_x = gx_ptrs,
            .grad_y = gy_ptrs,
            .grad_z = gz_ptrs,
            .phi    = lim_ptrs
        };

        for (std::size_t f = 0; f < n_inner; ++f) {
            const auto owner = static_cast<std::size_t>(mesh.face_owner[f]);
            const auto neigh = static_cast<std::size_t>(mesh.face_neigh[f]);

            const double area = mesh.face_area[f];
            const double nx   = mesh.face_normal_x[f];
            const double ny   = mesh.face_normal_y[f];
            const double nz   = mesh.face_normal_z[f];

            const double dx = dist_x[f];
            const double dy = dist_y[f];
            const double dz = dist_z[f];

            // Projection of centroid-to-centroid vector onto face normal
            const double proj = nx * dx + ny * dy + nz * dz;
            const double safe_proj = (proj > 1.0e-14) ? proj : 1.0e-14;

            // 1. Orthogonal Diffusion (Over-relaxed Jasak)
            const double delta_factor = area / safe_proj;
            const double D_f = mu * delta_factor;

            // Non-orthogonal decomposition vector: k_f = S_f - Delta_f
            const double kx = area * nx - delta_factor * dx;
            const double ky = area * ny - delta_factor * dy;
            const double kz = area * nz - delta_factor * dz;

            // 2. Implicit Convection (Upwind Difference Scheme - UDS)
            const double mdot = m_dot[f];
            const double mdot_pos = (mdot > 0.0) ? mdot : 0.0;
            const double mdot_neg = (mdot < 0.0) ? -mdot : 0.0; // max(-mdot, 0)

            // Matrix coefficients
            const double a_pn = -(D_f + mdot_neg); // Upper: coeff of neigh in owner eq
            const double a_np = -(D_f + mdot_pos); // Lower: coeff of owner in neigh eq

            mat.upper[f] = a_pn;
            mat.lower[f] = a_np;

            mat.diag[owner] += (D_f + mdot_pos);
            mat.diag[neigh] += (D_f + mdot_neg);

            // 3. Explicit Non-Orthogonal Diffusion Source (RHS)
            const double w_own = weight[f];
            const double w_ngh = 1.0 - w_own;

            const double gu_xf = w_own * fields.gu_x[owner] + w_ngh * fields.gu_x[neigh];
            const double gu_yf = w_own * fields.gu_y[owner] + w_ngh * fields.gu_y[neigh];
            const double gu_zf = w_own * fields.gu_z[owner] + w_ngh * fields.gu_z[neigh];
            const double diff_non_ortho_u = mu * (gu_xf * kx + gu_yf * ky + gu_zf * kz);

            const double gv_xf = w_own * fields.gv_x[owner] + w_ngh * fields.gv_x[neigh];
            const double gv_yf = w_own * fields.gv_y[owner] + w_ngh * fields.gv_y[neigh];
            const double gv_zf = w_own * fields.gv_z[owner] + w_ngh * fields.gv_z[neigh];
            const double diff_non_ortho_v = mu * (gv_xf * kx + gv_yf * ky + gv_zf * kz);

            const double gw_xf = w_own * fields.gw_x[owner] + w_ngh * fields.gw_x[neigh];
            const double gw_yf = w_own * fields.gw_y[owner] + w_ngh * fields.gw_y[neigh];
            const double gw_zf = w_own * fields.gw_z[owner] + w_ngh * fields.gw_z[neigh];
            const double diff_non_ortho_w = mu * (gw_xf * kx + gw_yf * ky + gw_zf * kz);

            mat.rhs_u[owner] += diff_non_ortho_u;
            mat.rhs_u[neigh] -= diff_non_ortho_u;

            mat.rhs_v[owner] += diff_non_ortho_v;
            mat.rhs_v[neigh] -= diff_non_ortho_v;

            mat.rhs_w[owner] += diff_non_ortho_w;
            mat.rhs_w[neigh] -= diff_non_ortho_w;

            // 4. Deferred Correction for Convection (High-Order - UDS -> RHS)
            if constexpr (ReconPolicy::kNeedsGradients) {
                double qL[3];
                double qR[3];
                ReconPolicy::template face_states<3>(batch, mesh, aux_geom, f, owner, neigh, qL, qR);

                const double u_ho = (mdot >= 0.0) ? qL[0] : qR[0];
                const double v_ho = (mdot >= 0.0) ? qL[1] : qR[1];
                const double w_ho = (mdot >= 0.0) ? qL[2] : qR[2];

                const double u_uds = (mdot >= 0.0) ? fields.u[owner] : fields.u[neigh];
                const double v_uds = (mdot >= 0.0) ? fields.v[owner] : fields.v[neigh];
                const double w_uds = (mdot >= 0.0) ? fields.w[owner] : fields.w[neigh];

                const double corr_u = mdot * (u_ho - u_uds);
                const double corr_v = mdot * (v_ho - v_uds);
                const double corr_w = mdot * (w_ho - w_uds);

                mat.rhs_u[owner] -= corr_u;
                mat.rhs_u[neigh] += corr_u;

                mat.rhs_v[owner] -= corr_v;
                mat.rhs_v[neigh] += corr_v;

                mat.rhs_w[owner] -= corr_w;
                mat.rhs_w[neigh] += corr_w;
            }
        }
    }

    /**
     * @brief Phase 2: Finalizes diagonal and RHS over cells.
     * Incorporates pressure gradients, unsteady terms, Patankar under-relaxation,
     * and extracts the Rhie-Chow coefficient d_P = V_P / a_P.
     * 
     * @note MUST be invoked AFTER boundary conditions have contributed to diag & RHS!
     * 
     * @param[in]     fields   Flow field views.
     * @param[in,out] mat      Matrix and RHS views.
     * @param[in]     rho      Fluid density.
     * @param[in]     alpha_u  Momentum under-relaxation factor (e.g. 0.7).
     * @param[in]     dt       Time step size (used only if TimeMode is unsteady).
     * @param[in]     mesh     Local mesh.
     */
    static void finalize_cells(
        const MomentumFieldViews& fields,
        MomentumMatrixViews& mat,
        const double rho,
        const double alpha_u,
        const double dt,
        const mesh::MeshPart& mesh) noexcept {

        const std::size_t n_own = static_cast<std::size_t>(mesh.n_own);
        const double inv_alpha = 1.0 / alpha_u;
        const double relax_factor = (1.0 - alpha_u) * inv_alpha;

        for (std::size_t c = 0; c < n_own; ++c) {
            const double vol = mesh.cell_volume[c];

            // 1. Pressure gradient source term: - V_c * grad(p)
            mat.rhs_u[c] -= vol * fields.gpx[c];
            mat.rhs_v[c] -= vol * fields.gpy[c];
            mat.rhs_w[c] -= vol * fields.gpz[c];

            // 2. Unsteady term: rho * V / dt
            if constexpr (kIsUnsteady) {
                assert(dt > 0.0 && "Time step dt must be positive for UnsteadyMode");
                const double a_time = (rho * vol) / dt;

                mat.diag[c]  += a_time;
                mat.rhs_u[c] += a_time * fields.u_old[c];
                mat.rhs_v[c] += a_time * fields.v_old[c];
                mat.rhs_w[c] += a_time * fields.w_old[c];
            }

            // 3. Patankar Under-Relaxation
            const double a_p_orig = mat.diag[c];

            mat.rhs_u[c] += relax_factor * a_p_orig * fields.u[c];
            mat.rhs_v[c] += relax_factor * a_p_orig * fields.v[c];
            mat.rhs_w[c] += relax_factor * a_p_orig * fields.w[c];

            const double a_p_relaxed = a_p_orig * inv_alpha;
            mat.diag[c] = a_p_relaxed;

            // 4. Rhie-Chow momentum coefficient: d = V_cell / a_P
            if (mat.d_coeff != nullptr) {
                mat.d_coeff[c] = vol / a_p_relaxed;
            }
        }
    }
};

} // namespace cfd::solver::incompressible
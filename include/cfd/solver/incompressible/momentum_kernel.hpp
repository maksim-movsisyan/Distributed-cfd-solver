#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"

namespace cfd::solver::incompressible {

/**
 * @class MomentumKernel
 * @brief High-performance momentum operator assembler for collocated unstructured FVM.
 *
 * @tparam TimeMode    SteadyMode or UnsteadyMode<Order> policy.
 * @tparam ReconPolicy Reconstruction policy (e.g. Muscl<LimiterPolicy>).
 */
template <typename TimeMode, typename ReconPolicy>
class MomentumKernel {
public:
    static constexpr bool kIsUnsteady = TimeMode::kIsUnsteady;
    static constexpr int kBdfOrder = TimeMode::kBdfOrder;

    MomentumKernel(const mesh::MeshPart& mesh, 
                   const mesh::MeshAuxConnectivity& aux_conn,
                   const mesh::MeshAuxGeometry& aux_geom)
        : mesh_(mesh), aux_conn_(aux_conn), aux_geom_(aux_geom) {}
    
    /**
     * @brief Phase 1: Assembles interior face fluxes into raw LDU matrix and RHS arrays.
     *
     * @param[in]  q        Span of NVars pointers to cell primitive variables (SoA).
     * @param[in]  grad_x   Span of NVars pointers to d/dx gradient components.
     * @param[in]  grad_y   Span of NVars pointers to d/dy gradient components.
     * @param[in]  grad_z   Span of NVars pointers to d/dz gradient components.
     * @param[in]  phi      Span of NVars pointers to limiters.
     * @param[in]  m_dot    Mass fluxes at faces.
     * @param[out] diag     LDU main diagonal array [n_own].
     * @param[out] upper    LDU upper triangle array [n_inner_faces].
     * @param[out] lower    LDU lower triangle array [n_inner_faces].
     * @param[out] rhs_u    Right-hand side for u-momentum [n_own].
     * @param[out] rhs_v    Right-hand side for v-momentum [n_own].
     * @param[out] rhs_w    Right-hand side for w-momentum [n_own].
     * @param[in]  mu       Dynamic viscosity (Pa*s).
     * @param[in]  mut      Eddy viscosity (Pa*s) [n_cells].
     */
    void apply(std::span<const double* const> q,
               std::span<const double* const> grad_x,
               std::span<const double* const> grad_y,
               std::span<const double* const> grad_z,
               std::span<const double* const> phi,
               const double* CFD_RESTRICT m_dot,
               double* CFD_RESTRICT diag,
               double* CFD_RESTRICT upper,
               double* CFD_RESTRICT lower,
               double* CFD_RESTRICT rhs_u,
               double* CFD_RESTRICT rhs_v,
               double* CFD_RESTRICT rhs_w,
               const double mu,
               const double* CFD_RESTRICT mut = nullptr) const noexcept {

        const std::size_t n_inner = static_cast<std::size_t>(mesh_.n_inner_faces);

        const double* CFD_RESTRICT dist_x = aux_geom_.face_cell_dist_x.data();
        const double* CFD_RESTRICT dist_y = aux_geom_.face_cell_dist_y.data();
        const double* CFD_RESTRICT dist_z = aux_geom_.face_cell_dist_z.data();
        const double* CFD_RESTRICT weight = aux_geom_.face_interp_weight.data();

        // Batch representation for 3-component face reconstruction (u, v, w)
        const double* CFD_RESTRICT q_ptrs[3]   = { q[1], q[2], q[3] };
        const double* CFD_RESTRICT gx_ptrs[3]  = { grad_x[1], grad_x[2], grad_x[3] };
        const double* CFD_RESTRICT gy_ptrs[3]  = { grad_y[1], grad_y[2], grad_y[3] };
        const double* CFD_RESTRICT gz_ptrs[3]  = { grad_z[1], grad_z[2], grad_z[3] };
        const double* CFD_RESTRICT lim_ptrs[3] = { phi[1], phi[2], phi[3] };
                
        const numerics::recon::ReconBatchField<3> batch(
            q_ptrs, gx_ptrs, gy_ptrs, gz_ptrs, lim_ptrs
        );

        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t owner = static_cast<std::size_t>(mesh_.face_owner[f]);
            const std::size_t neigh = static_cast<std::size_t>(mesh_.face_neigh[f]);
            
            const double mutL = mut ? mut[owner] : 0.0;
            const double mutR = mut ? mut[neigh] : 0.0;
            const double mu_eff = 0.5 * (mutL + mutR) + mu;

            const double area = mesh_.face_area[f];
            const double nx   = mesh_.face_normal_x[f];
            const double ny   = mesh_.face_normal_y[f];
            const double nz   = mesh_.face_normal_z[f];

            const double dx = dist_x[f];
            const double dy = dist_y[f];
            const double dz = dist_z[f];

            // Projection of centroid-to-centroid vector onto unit face normal
            const double proj = nx * dx + ny * dy + nz * dz;
            const double safe_proj = (proj > 1.0e-14) ? proj : 1.0e-14;

            // 1. Orthogonal Diffusion (Over-relaxed Jasak decomposition)
            const double delta_factor = area / safe_proj;
            const double D_f = mu_eff * delta_factor;

            // Non-orthogonal decomposition vector: k_f = S_f - Delta_f
            const double kx = area * nx - delta_factor * dx;
            const double ky = area * ny - delta_factor * dy;
            const double kz = area * nz - delta_factor * dz;

            // 2. Implicit Convection (Upwind Difference Scheme)
            const double mdot     = m_dot[f];
            const double mdot_pos = (mdot > 0.0) ? mdot : 0.0;
            const double mdot_neg = (mdot < 0.0) ? -mdot : 0.0;

            // Raw LDU matrix entries
            const double a_pn = -(D_f + mdot_neg); // Upper: neighbor in owner eq
            const double a_np = -(D_f + mdot_pos); // Lower: owner in neighbor eq

            upper[f] = a_pn;
            lower[f] = a_np;

            diag[owner] += (D_f + mdot_pos);
            diag[neigh] += (D_f + mdot_neg);

            // 3. Explicit Non-Orthogonal Diffusion Source (RHS)
            const double w_own = weight[f];
            const double w_ngh = 1.0 - w_own;

            const double gu_xf = w_own * gx_ptrs[0][owner] + w_ngh * gx_ptrs[0][neigh];
            const double gu_yf = w_own * gy_ptrs[0][owner] + w_ngh * gy_ptrs[0][neigh];
            const double gu_zf = w_own * gz_ptrs[0][owner] + w_ngh * gz_ptrs[0][neigh];
            const double diff_non_ortho_u = mu_eff * (gu_xf * kx + gu_yf * ky + gu_zf * kz);

            const double gv_xf = w_own * gx_ptrs[1][owner] + w_ngh * gx_ptrs[1][neigh];
            const double gv_yf = w_own * gy_ptrs[1][owner] + w_ngh * gy_ptrs[1][neigh];
            const double gv_zf = w_own * gz_ptrs[1][owner] + w_ngh * gz_ptrs[1][neigh];
            const double diff_non_ortho_v = mu_eff * (gv_xf * kx + gv_yf * ky + gv_zf * kz);

            const double gw_xf = w_own * gx_ptrs[2][owner] + w_ngh * gx_ptrs[2][neigh];
            const double gw_yf = w_own * gy_ptrs[2][owner] + w_ngh * gy_ptrs[2][neigh];
            const double gw_zf = w_own * gz_ptrs[2][owner] + w_ngh * gz_ptrs[2][neigh];
            const double diff_non_ortho_w = mu_eff * (gw_xf * kx + gw_yf * ky + gw_zf * kz);

            rhs_u[owner] += diff_non_ortho_u;
            rhs_u[neigh] -= diff_non_ortho_u;

            rhs_v[owner] += diff_non_ortho_v;
            rhs_v[neigh] -= diff_non_ortho_v;

            rhs_w[owner] += diff_non_ortho_w;
            rhs_w[neigh] -= diff_non_ortho_w;

            // 4. Deferred Correction for Convection (High-Order - UDS -> RHS)
            if constexpr (ReconPolicy::kNeedsGradients) {
                double qL[3];
                double qR[3];
                ReconPolicy::template face_states<3>(batch, mesh_, aux_geom_, f, owner, neigh, qL, qR);

                const double u_ho = (mdot >= 0.0) ? qL[0] : qR[0];
                const double v_ho = (mdot >= 0.0) ? qL[1] : qR[1];
                const double w_ho = (mdot >= 0.0) ? qL[2] : qR[2];

                const double u_uds = (mdot >= 0.0) ? q_ptrs[0][owner] : q_ptrs[0][neigh];
                const double v_uds = (mdot >= 0.0) ? q_ptrs[1][owner] : q_ptrs[1][neigh];
                const double w_uds = (mdot >= 0.0) ? q_ptrs[2][owner] : q_ptrs[2][neigh];

                const double corr_u = mdot * (u_ho - u_uds);
                const double corr_v = mdot * (v_ho - v_uds);
                const double corr_w = mdot * (w_ho - w_uds);

                rhs_u[owner] -= corr_u;
                rhs_u[neigh] += corr_u;

                rhs_v[owner] -= corr_v;
                rhs_v[neigh] += corr_v;

                rhs_w[owner] -= corr_w;
                rhs_w[neigh] += corr_w;
            }
        }
    }

    /**
     * @brief Phase 2: Finalizes diagonal and RHS across mesh cells.
     * Incorporates pressure gradients, unsteady time schemes (BDF1 / BDF2),
     * Patankar under-relaxation, and computes Rhie-Chow coefficient d_P = V_P / a_P.
     *
     * @note MUST be called AFTER boundary conditions have contributed to diag & RHS!
     */
    void finalize(std::span<const double* const> q,
                  std::span<const double* const> q_old,
                  std::span<const double* const> q_older,
                  std::span<const double* const> grad_x,
                  std::span<const double* const> grad_y,
                  std::span<const double* const> grad_z,
                  double* CFD_RESTRICT diag,
                  double* CFD_RESTRICT rhs_u,
                  double* CFD_RESTRICT rhs_v,
                  double* CFD_RESTRICT rhs_w,
                  double* CFD_RESTRICT d_coeff,
                  const double rho,
                  const double alpha_u,
                  const double dt = 0.0) const noexcept {

        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);
        const double inv_alpha = 1.0 / alpha_u;
        const double relax_factor = (1.0 - alpha_u) * inv_alpha;

        const double* CFD_RESTRICT prs_grad_x = grad_x[0];
        const double* CFD_RESTRICT prs_grad_y = grad_y[0];
        const double* CFD_RESTRICT prs_grad_z = grad_z[0];

        for (std::size_t c = 0; c < n_own; ++c) {
            const double vol = mesh_.cell_volume[c];

            // 1. Pressure gradient source term: - V_c * grad(p)
            rhs_u[c] -= vol * prs_grad_x[c];
            rhs_v[c] -= vol * prs_grad_y[c];
            rhs_w[c] -= vol * prs_grad_z[c];

            // 2. Unsteady term dispatch (Steady / BDF1 / BDF2)
            if constexpr (kIsUnsteady) {
                assert(dt > 0.0 && "Time step dt must be positive for UnsteadyMode");
                const double coeff = (rho * vol) / dt;

                if constexpr (kBdfOrder == 1) {
                    // BDF1 (Backward Euler): a_time = rho * V / dt
                    const double a_time = coeff;
                    diag[c]  += a_time;
                    rhs_u[c] += a_time * q_old[1][c];
                    rhs_v[c] += a_time * q_old[2][c];
                    rhs_w[c] += a_time * q_old[3][c];

                } else if constexpr (kBdfOrder == 2) {
                    assert(q_older.size() >= 4 && q_older[1] && q_older[2] && q_older[3] &&
                        "BDF2 requires t^{n-1} velocity fields");

                    const double a_time = 1.5 * coeff;
                    diag[c]  += a_time;
                    rhs_u[c] += coeff * (2.0 * q_old[1][c] - 0.5 * q_older[1][c]);
                    rhs_v[c] += coeff * (2.0 * q_old[2][c] - 0.5 * q_older[2][c]);
                    rhs_w[c] += coeff * (2.0 * q_old[3][c] - 0.5 * q_older[3][c]);
                }
            }

            // 3. Patankar Under-Relaxation
            const double a_p_orig = diag[c];

            rhs_u[c] += relax_factor * a_p_orig * q[1][c];
            rhs_v[c] += relax_factor * a_p_orig * q[2][c];
            rhs_w[c] += relax_factor * a_p_orig * q[3][c];

            const double a_p_relaxed = a_p_orig * inv_alpha;
            diag[c] = a_p_relaxed;

            // 4. Rhie-Chow momentum coefficient: d = V_cell / a_P
            if (d_coeff != nullptr) {
                d_coeff[c] = vol / a_p_relaxed;
            }
        }
    }

private:
    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    const mesh::MeshAuxGeometry& aux_geom_;
};

} // namespace cfd::solver::incompressible
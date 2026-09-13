// Second-order Navier-Stokes viscous diffusion flux with directional over-relaxed
// non-orthogonal correction and Sutherland/Prandtl transport property models.
//
// Zero-overhead policy struct designed to mirror the HllcFlux interface.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"

namespace cfd::solver::compressible::fluxes {

/**
 * @struct ViscousViews
 * @brief Zero-overhead bundle of raw pointers to primitive and gradient SoA arrays.
 * Initialized ONCE per residual evaluation outside face loops.
 */
struct ViscousViews {
    const double* CFD_RESTRICT prs;
    const double* CFD_RESTRICT u;
    const double* CFD_RESTRICT v;
    const double* CFD_RESTRICT w;
    const double* CFD_RESTRICT T;

    const double* CFD_RESTRICT du_dx;
    const double* CFD_RESTRICT du_dy;
    const double* CFD_RESTRICT du_dz;

    const double* CFD_RESTRICT dv_dx;
    const double* CFD_RESTRICT dv_dy;
    const double* CFD_RESTRICT dv_dz;

    const double* CFD_RESTRICT dw_dx;
    const double* CFD_RESTRICT dw_dy;
    const double* CFD_RESTRICT dw_dz;

    const double* CFD_RESTRICT dT_dx;
    const double* CFD_RESTRICT dT_dy;
    const double* CFD_RESTRICT dT_dz;

    static ViscousViews from_spans(std::span<const double* const> q,
                                  std::span<const double* const> gx,
                                  std::span<const double* const> gy,
                                  std::span<const double* const> gz) noexcept {
        return ViscousViews{
            q[0], q[1], q[2], q[3], q[4],
            gx[1], gy[1], gz[1],
            gx[2], gy[2], gz[2],
            gx[3], gy[3], gz[3],
            gx[4], gy[4], gz[4]
        };
    }
};

struct ViscousFlux {
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::FaceCellDistanceInv|
                                                      mesh::AuxGeomType::FaceCellDistanceVector; 
    [[nodiscard]] static constexpr const char* name() noexcept { return "VISCOUS"; }

    // =========================================================================
    // 1. Transport Properties
    // =========================================================================

    /** @brief Sutherland dynamic viscosity [Pa s]. */
    [[nodiscard]] static inline double viscosity(const double T) noexcept {
        constexpr double kSuthConst = constants::kSutherlandViscosity 
                                    * (constants::kSutherlandTRef + constants::kSutherlandT);
        const double ratio = T / constants::kSutherlandTRef;
        return kSuthConst * ratio * std::sqrt(ratio) / (T + constants::kSutherlandT);
    }

    /** @brief Constant Prandtl thermal conductivity [W / (m K)]. */
    template <eos::EquationOfStatePolicy EOS>
    [[nodiscard]] static inline double thermal_conductivity(const EOS& eos, 
                                                            const double T, const double p, 
                                                            const double prandtl) noexcept {
        return (viscosity(T) * eos.cp_Tp(T, p)) / prandtl;
    }

    // =========================================================================
    // 2. Viscous Face Flux Evaluation (Flat SoA Gradient Layout)
    // =========================================================================
    /**
     * @brief Evaluates viscous flux using direct SoA memory loads with zero intermediate stack copies.
     */
    template <eos::EquationOfStatePolicy EOS>
    static inline void face_flux(const EOS& eos,
                                 const mesh::MeshAuxGeometry& aux_geom,
                                 const std::size_t f,
                                 const std::size_t c0,
                                 const std::size_t c1,
                                 const ViscousViews& v,
                                 const double mutL, const double mutR,
                                 const double nx, const double ny, const double nz,
                                 const double area,
                                 double Fv[5],
                                 double& lam_visc,
                                 const double prandtl_lam, 
                                 const double prandtl_turb) noexcept {
        // Face-averaged primitive values (direct memory load into registers)
        const double p_f = 0.5 * (v.prs[c0] + v.prs[c1]);
        const double u_f = 0.5 * (v.u[c0]   + v.u[c1]);
        const double v_f = 0.5 * (v.v[c0]   + v.v[c1]);
        const double w_f = 0.5 * (v.w[c0]   + v.w[c1]);
        const double T_f = 0.5 * (v.T[c0]   + v.T[c1]);

        // Effective transport properties
        const double mu_lam = viscosity(T_f);
        const double mu_t_f = 0.5 * (mutL + mutR);
        const double mu_eff = mu_lam + mu_t_f;

        const double k_lam = thermal_conductivity(eos, T_f, p_f, prandtl_lam);
        const double cp    = eos.cp_Tp(T_f, p_f);
        const double k_eff = k_lam + (mu_t_f * cp) / prandtl_turb;

        // Geometry & Non-orthogonal correction
        const double inv_d = aux_geom.face_cell_dist_inv[f];
        const double xi_x  = aux_geom.face_cell_dist_x[f] * inv_d;
        const double xi_y  = aux_geom.face_cell_dist_y[f] * inv_d;
        const double xi_z  = aux_geom.face_cell_dist_z[f] * inv_d;

        const double n_corr_x = nx - xi_x;
        const double n_corr_y = ny - xi_y;
        const double n_corr_z = nz - xi_z;

        // Face-averaged spatial derivatives
        const double du_dx = 0.5 * (v.du_dx[c0] + v.du_dx[c1]);
        const double du_dy = 0.5 * (v.du_dy[c0] + v.du_dy[c1]);
        const double du_dz = 0.5 * (v.du_dz[c0] + v.du_dz[c1]);

        const double dv_dx = 0.5 * (v.dv_dx[c0] + v.dv_dx[c1]);
        const double dv_dy = 0.5 * (v.dv_dy[c0] + v.dv_dy[c1]);
        const double dv_dz = 0.5 * (v.dv_dz[c0] + v.dv_dz[c1]);

        const double dw_dx = 0.5 * (v.dw_dx[c0] + v.dw_dx[c1]);
        const double dw_dy = 0.5 * (v.dw_dy[c0] + v.dw_dy[c1]);
        const double dw_dz = 0.5 * (v.dw_dz[c0] + v.dw_dz[c1]);

        const double dT_dx = 0.5 * (v.dT_dx[c0] + v.dT_dx[c1]);
        const double dT_dy = 0.5 * (v.dT_dy[c0] + v.dT_dy[c1]);
        const double dT_dz = 0.5 * (v.dT_dz[c0] + v.dT_dz[c1]);

        const double div_v = du_dx + dv_dy + dw_dz;
        const double two_thirds_div_v = (2.0 / 3.0) * div_v;

        // Over-relaxed normal derivatives
        const double du_dn = (v.u[c1] - v.u[c0]) * inv_d + (du_dx * n_corr_x + du_dy * n_corr_y + du_dz * n_corr_z);
        const double dv_dn = (v.v[c1] - v.v[c0]) * inv_d + (dv_dx * n_corr_x + dv_dy * n_corr_y + dv_dz * n_corr_z);
        const double dw_dn = (v.w[c1] - v.w[c0]) * inv_d + (dw_dx * n_corr_x + dw_dy * n_corr_y + dw_dz * n_corr_z);
        const double dT_dn = (v.T[c1] - v.T[c0]) * inv_d + (dT_dx * n_corr_x + dT_dy * n_corr_y + dT_dz * n_corr_z);

        // Viscous normal stress vector: tau_n = tau · n
        const double tau_nx = mu_eff * (du_dn + (du_dx * nx + dv_dx * ny + dw_dx * nz) - two_thirds_div_v * nx);
        const double tau_ny = mu_eff * (dv_dn + (du_dy * nx + dv_dy * ny + dw_dy * nz) - two_thirds_div_v * ny);
        const double tau_nz = mu_eff * (dw_dn + (du_dz * nx + dv_dz * ny + dw_dz * nz) - two_thirds_div_v * nz);

        // Heat flux normal component  
        const double qn = -k_eff * dT_dn;

        // Viscous spectral radius contribution: (4/3) * nu_eff * |d|^-1 * area
        const double rho_f = eos.density_Tp(T_f, p_f);
        const double nu_eff = mu_eff / rho_f;
        lam_visc = (4.0 / 3.0) * nu_eff * inv_d * area;

        // Assembly of viscous flux vector Fv (multiplied by area)
        Fv[0] = 0.0;
        Fv[1] = -tau_nx * area;
        Fv[2] = -tau_ny * area;
        Fv[3] = -tau_nz * area;
        Fv[4] = (-(u_f * tau_nx + v_f * tau_ny + w_f * tau_nz) + qn) * area;
    }

    // =========================================================================
    // 3. Viscous Face Flux Evaluation (Cell-Local Gradients, e.g. for BCs)
    // =========================================================================
    /**
     * @brief Face-local overload accepting explicit cell gradient vectors (size 12: [du, dv, dw, dT]).
     *        Useful for boundary conditions or local ghost reconstruction sweeps.
     */
    template <eos::EquationOfStatePolicy EOS>
    static inline void face_flux(const EOS& eos,
                                 const mesh::MeshAuxGeometry& aux_geom,
                                 const std::size_t f,
                                 const double qL[5],
                                 const double qR[5],
                                 const double gradL[12],
                                 const double gradR[12],
                                 const double mutL, const double mutR,
                                 const double nx, const double ny, const double nz,
                                 const double area,
                                 double Fv[5],
                                 double& lam_visc,
                                 const double prandtl_lam, const double prandtl_turb) noexcept {
        const double p_f = 0.5 * (qL[0] + qR[0]);
        const double u_f = 0.5 * (qL[1] + qR[1]);
        const double v_f = 0.5 * (qL[2] + qR[2]);
        const double w_f = 0.5 * (qL[3] + qR[3]);
        const double T_f = 0.5 * (qL[4] + qR[4]);

        const double mu_lam = viscosity(T_f);
        const double mu_t_f = 0.5 * (mutL + mutR);
        const double mu_eff = mu_lam + mu_t_f;

        const double k_lam = thermal_conductivity(eos, T_f, p_f, prandtl_lam);
        const double cp    = eos.cp_Tp(T_f, p_f);
        const double k_eff = k_lam + (mu_t_f * cp) / prandtl_turb;

        const double inv_d = aux_geom.face_cell_dist_inv[f];
        const double xi_x  = aux_geom.face_cell_dist_x[f] * inv_d;
        const double xi_y  = aux_geom.face_cell_dist_y[f] * inv_d;
        const double xi_z  = aux_geom.face_cell_dist_z[f] * inv_d;

        const double n_corr_x = nx - xi_x;
        const double n_corr_y = ny - xi_y;
        const double n_corr_z = nz - xi_z;

        // grad indices: 0..2 (u), 3..5 (v), 6..8 (w), 9..11 (T)
        const double du_dx = 0.5 * (gradL[0] + gradR[0]);
        const double du_dy = 0.5 * (gradL[1] + gradR[1]);
        const double du_dz = 0.5 * (gradL[2] + gradR[2]);

        const double dv_dx = 0.5 * (gradL[3] + gradR[3]);
        const double dv_dy = 0.5 * (gradL[4] + gradR[4]);
        const double dv_dz = 0.5 * (gradL[5] + gradR[5]);

        const double dw_dx = 0.5 * (gradL[6] + gradR[6]);
        const double dw_dy = 0.5 * (gradL[7] + gradR[7]);
        const double dw_dz = 0.5 * (gradL[8] + gradR[8]);

        const double dT_dx = 0.5 * (gradL[9]  + gradR[9]);
        const double dT_dy = 0.5 * (gradL[10] + gradR[10]);
        const double dT_dz = 0.5 * (gradL[11] + gradR[11]);

        const double div_v = du_dx + dv_dy + dw_dz;
        const double two_thirds_div_v = (2.0 / 3.0) * div_v;

        const double du_dn = (qR[1] - qL[1]) * inv_d + (du_dx * n_corr_x + du_dy * n_corr_y + du_dz * n_corr_z);
        const double dv_dn = (qR[2] - qL[2]) * inv_d + (dv_dx * n_corr_x + dv_dy * n_corr_y + dv_dz * n_corr_z);
        const double dw_dn = (qR[3] - qL[3]) * inv_d + (dw_dx * n_corr_x + dw_dy * n_corr_y + dw_dz * n_corr_z);
        const double dT_dn = (qR[4] - qL[4]) * inv_d + (dT_dx * n_corr_x + dT_dy * n_corr_y + dT_dz * n_corr_z);

        const double tau_nx = mu_eff * (du_dn + (du_dx * nx + dv_dx * ny + dw_dx * nz) - two_thirds_div_v * nx);
        const double tau_ny = mu_eff * (dv_dn + (du_dy * nx + dv_dy * ny + dw_dy * nz) - two_thirds_div_v * ny);
        const double tau_nz = mu_eff * (dw_dn + (du_dz * nx + dv_dz * ny + dw_dz * nz) - two_thirds_div_v * nz);

        const double qn = -k_eff * dT_dn;

        const double rho_f = eos.density_Tp(T_f, p_f);
        const double nu_eff = mu_eff / rho_f;
        lam_visc = (4.0 / 3.0) * nu_eff * inv_d * area;

        Fv[0] = 0.0;
        Fv[1] = -tau_nx * area;
        Fv[2] = -tau_ny * area;
        Fv[3] = -tau_nz * area;
        Fv[4] = (-(u_f * tau_nx + v_f * tau_ny + w_f * tau_nz) + qn) * area;
    }

    // =========================================================================
    // 4. Analytical Thin-Layer Viscous Jacobian (LHS Matrix Assembly)
    // =========================================================================
    /**
     * @brief Computes analytical thin-layer viscous flux Jacobians dFv/dUL and dFv/dUR (5x5, row-major).
     *
     * Linearizes the directional normal diffusion flux along the line of cell centroids:
     *   Fv ≈ J_visc · (UL - UR)
     *   dFv/dUL = +Area * J_visc
     *   dFv/dUR = -Area * J_visc
     *
     * Mathematical properties:
     *  - Strictly positive diagonal entries (+Area * mu/rho, +Area * k/(rho*cv)) guaranteeing
     *    unconditional diagonal dominance and robust ILU preconditioning;
     *  - Skew-symmetric coupling between adjacent cells (Mtrx_R = -Mtrx_L);
     *  - Directly utilizes EOS thermodynamic functions without artificial temperature substitutions.
     */
    template <eos::EquationOfStatePolicy EOS>
    static inline void face_flux_jacobian(const EOS& eos,
                                          const mesh::MeshAuxGeometry& aux_geom,
                                          const std::size_t f,
                                          const double qL[5],
                                          const double qR[5],
                                          const double mutL,
                                          const double mutR,
                                          const double area,
                                          double dFL[25],
                                          double dFR[25],
                                          const double prandtl_lam, const double prandtl_turb) noexcept {
        constexpr int N = 5;

        // 1. Face-averaged state
        const double p_f = 0.5 * (qL[0] + qR[0]);
        const double u_f = 0.5 * (qL[1] + qR[1]);
        const double v_f = 0.5 * (qL[2] + qR[2]);
        const double w_f = 0.5 * (qL[3] + qR[3]);
        const double T_f = 0.5 * (qL[4] + qR[4]);

        const double rho_f = eos.density_Tp(T_f, p_f);
        const double inv_rho = 1.0 / rho_f;
        const double cv = eos.cv_Tp(T_f, p_f);
        const double inv_rho_cv = 1.0 / (rho_f * cv);

        // 2. Effective transport coefficients
        const double mu_lam = viscosity(T_f);
        const double mu_t_f = 0.5 * (mutL + mutR);
        const double mu_eff = mu_lam + mu_t_f;

        const double k_lam = thermal_conductivity(eos, T_f, p_f, prandtl_lam);
        const double cp    = eos.cp_Tp(T_f, p_f);
        const double k_eff = k_lam + (mu_t_f * cp) / prandtl_turb;

        // 3. Metric scaling: C = coeff * Area / |d|
        const double inv_d = aux_geom.face_cell_dist_inv[f];
        const double C_mu  = mu_eff * inv_d * area;
        const double C_lam = k_eff  * inv_d * area;

        const double V2 = u_f * u_f + v_f * v_f + w_f * w_f;

        // 4. Assemble 5x5 Thin-Layer Jacobian Matrix J_visc
        // Row 0: Continuity (Viscosity does not transport mass)
        dFL[0] = 0.0; dFL[1] = 0.0; dFL[2] = 0.0; dFL[3] = 0.0; dFL[4] = 0.0;

        // Row 1: X-Momentum
        dFL[5] = -C_mu * u_f * inv_rho;
        dFL[6] =  C_mu * inv_rho;
        dFL[7] =  0.0;
        dFL[8] =  0.0;
        dFL[9] =  0.0;

        // Row 2: Y-Momentum
        dFL[10] = -C_mu * v_f * inv_rho;
        dFL[11] =  0.0;
        dFL[12] =  C_mu * inv_rho;
        dFL[13] =  0.0;
        dFL[14] =  0.0;

        // Row 3: Z-Momentum
        dFL[15] = -C_mu * w_f * inv_rho;
        dFL[16] =  0.0;
        dFL[17] =  0.0;
        dFL[18] =  C_mu * inv_rho;
        dFL[19] =  0.0;

        // Row 4: Total Energy (Heat Conduction + Viscous Dissipation Work)
        // Conduction: C_lam * dT/dU
        // Work: u * (Row 1) + v * (Row 2) + w * (Row 3)
        dFL[20] = C_lam * (0.5 * V2 - cv * T_f) * inv_rho_cv - C_mu * V2 * inv_rho;
        dFL[21] = -C_lam * u_f * inv_rho_cv + C_mu * u_f * inv_rho;
        dFL[22] = -C_lam * v_f * inv_rho_cv + C_mu * v_f * inv_rho;
        dFL[23] = -C_lam * w_f * inv_rho_cv + C_mu * w_f * inv_rho;
        dFL[24] =  C_lam * inv_rho_cv;

        // 5. Symmetric inter-cell coupling: dFv/dUR = -dFv/dUL
        for (int i = 0; i < N * N; ++i) {
            dFR[i] = -dFL[i];
        }
    }
};

} // namespace cfd::solver::fluxes
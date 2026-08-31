// Residual evaluation R(U): the reusable functional behind every time
// integrator (explicit stages now, matrix/matrix-free implicit later).
//
// The kernel consumes PRIMITIVE cell states: spatial reconstruction (1st-order or MUSCL)
// is primitive-based, and the conserved states required by the numerical flux
// are converted per face from the reconstructed primitives. Viscous face
// fluxes are appended by the flow-physics policy (compiled out for Euler).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver {

/**
 * @class ResidualKernel
 * @brief High-performance cell-centered finite volume spatial residual evaluator.
 *
 * Accumulates total flux balance:
 *   Res = sum_{faces} (F_inv + F_visc) * Area
 * and cell spectral wave radii (inviscid + viscous stability estimates):
 *   Lambda = sum_{faces} (Area * s_max)
 *
 * Update rule in time integrator: U^{n+1} = U^n - (dt / V) * Res
 *
 * @tparam EOS   Thermodynamic Equation of State conforming to eos::EquationOfState
 * @tparam Flux  Numerical flux policy (e.g., riemann::HllcFlux)
 * @tparam Recon Spatial reconstruction policy (recon::FirstOrder, recon::Muscl<Limiter>)
 * @tparam Phys  Flow equation set
 */
template <eos::EquationOfState EOS, typename Flux,
          recon::ReconstructionPolicy Recon, physics::PhysicsGeneral Phys>
class ResidualKernel {
public:
    ResidualKernel(const mesh::MeshPart& mp, const EOS eos, const Phys phys)
        : mp_(mp), eos_(eos), phys_(phys), phys_geom_(Phys::build_geometry(mp)) {}

    /**
     * @brief Evaluates the spatial residual and spectral radius across the local partition.
     *
     * @param[in]  q     Primitive cell states [p, u, v, w, T], halo- and BC-ghost-complete.
     * @param[in]  grad  Cell gradients (valid if kNeedsGradients == true).
     * @param[in]  phi   Cell gradient limiters in [0, 1].
     * @param[in]  geom  Reconstruction precomputed geometry (from Recon::build_geometry).
     * @param[out] res   Accumulated flux balance vector.
     * @param[out] lam   Accumulated per-cell spectral radius (size >= n_cells).
     * @param[in]  mut   Eddy viscosity per cell (nullptr without turbulence).
     * @param[out] mdot  Face mass flux storage (nullptr unless physics modules request it).
     */
    void apply(fields::ConstPrimitiveView q,
               fields::ConstPrimitiveGradView grad,
               fields::ConstPrimitiveView phi,
               const typename Recon::Geometry& geom,
               fields::ResidualView<double> res,
               double* CFD_RESTRICT lam,
               const double* CFD_RESTRICT mut = nullptr,
               double* CFD_RESTRICT mdot = nullptr) const noexcept {
        const std::size_t n_inner  = static_cast<std::size_t>(mp_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mp_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mp_.n_cells);
        const std::size_t n_bfaces = n_faces - n_inner;
        const std::size_t n_total  = n_cells + n_bfaces;

        // 1. Reset residuals and spectral radii
        std::fill(res.res1, res.res1 + n_total, 0.0);
        std::fill(res.res2, res.res2 + n_total, 0.0);
        std::fill(res.res3, res.res3 + n_total, 0.0);
        std::fill(res.res4, res.res4 + n_total, 0.0);
        std::fill(res.res5, res.res5 + n_total, 0.0);
        std::fill(lam, lam + n_cells, 0.0);

        // 2. Fused flux evaluation (Euler Riemann solver + Navier-Stokes diffusion)
        compute_fluxes(q, grad, phi, geom, res, lam, mut, mdot);
    }

    /**
     * @brief Computes rank-local net mass and total energy flux through each boundary patch.
     */
    void boundary_integrals(fields::ConstPrimitiveView q,
                            fields::ConstPrimitiveGradView grad,
                            fields::ConstPrimitiveView phi,
                            const typename Recon::Geometry& geom,
                            std::vector<double>& mass,
                            std::vector<double>& energy,
                            const double* CFD_RESTRICT mut = nullptr) const noexcept {
        const std::size_t n_inner = static_cast<std::size_t>(mp_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mp_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mp_.n_cells);
        const std::size_t np      = mp_.patches.size();

        mass.assign(np, 0.0);
        energy.assign(np, 0.0);

        const LocalIndex* CFD_RESTRICT owner = mp_.face_owner.data();
        const mesh::PatchId* CFD_RESTRICT patch = mp_.face_patch.data();
        const double* CFD_RESTRICT nx = mp_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mp_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mp_.face_normal_z.data();
        const double* CFD_RESTRICT area = mp_.face_area.data();

        const recon::ReconField rf{q, grad, phi};

        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[Phys::kNumVars];
            double qR[Phys::kNumVars];
            Recon::boundary_face_states(rf, geom, f, c0, cg, qL, qR);

            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[Phys::kNumVars];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            double energy_flux = F[4];

            if constexpr (Phys::kHasViscous) {
                const auto visc = compute_face_viscous_flux(f, c0, cg, q, grad, mut, nxf, nyf, nzf, Af);
                energy_flux += visc.Fv4;
            }

            const std::size_t p = static_cast<std::size_t>(patch[f]);
            mass[p]   += F[0];
            energy[p] += energy_flux;
        }
    }

    [[nodiscard]] const EOS& eos() const noexcept { return eos_; }

    /** @brief Physics (mean-flow) geometry — consumed by module face sweeps. */
    [[nodiscard]] const typename Phys::Geometry& phys_geometry() const noexcept {
        return phys_geom_;
    }

private:
    struct ViscousFaceResult {
        double Fv1{0.0};
        double Fv2{0.0};
        double Fv3{0.0};
        double Fv4{0.0};
        double lam_visc{0.0};
    };

    /**
     * @brief Computes viscous diffusion flux and spectral radius contribution for a single face.
     */
    [[nodiscard]] inline ViscousFaceResult compute_face_viscous_flux(
        std::size_t f, std::size_t c0, std::size_t c1,
        fields::ConstPrimitiveView q,
        fields::ConstPrimitiveGradView grad,
        const double* CFD_RESTRICT mut,
        double nxf, double nyf, double nzf, double Af) const noexcept {
        const double* CFD_RESTRICT g_dx = phys_geom_.dx.data();
        const double* CFD_RESTRICT g_dy = phys_geom_.dy.data();
        const double* CFD_RESTRICT g_dz = phys_geom_.dz.data();
        const double* CFD_RESTRICT g_inv_d = phys_geom_.inv_d.data();

        const double pL = q.prs[c0], uL = q.vx[c0], vL = q.vy[c0], wL = q.vz[c0], TL = q.tmp[c0];
        const double pR = q.prs[c1], uR = q.vx[c1], vR = q.vy[c1], wR = q.vz[c1], TR = q.tmp[c1];

        const double u_f = 0.5 * (uL + uR);
        const double v_f = 0.5 * (vL + vR);
        const double w_f = 0.5 * (wL + wR);
        const double T_f = 0.5 * (TL + TR);
        const double p_f = 0.5 * (pL + pR);

        const double mu_lam = phys_.viscosity(T_f);
        const double mu_t_f = mut ? 0.5 * (mut[c0] + mut[c1]) : 0.0;
        const double mu_eff = mu_lam + mu_t_f;

        const double k_lam = phys_.thermal_conductivity(eos_, T_f, p_f);
        const double cp    = eos_.cp_Tp(T_f, p_f);
        constexpr double kInvPrt = 1.0 / constants::kTurbPrandtl;
        const double k_eff = k_lam + mu_t_f * cp * kInvPrt;

        const double inv_d = g_inv_d[f];
        const double xi_x  = g_dx[f] * inv_d;
        const double xi_y  = g_dy[f] * inv_d;
        const double xi_z  = g_dz[f] * inv_d;

        const double n_corr_x = nxf - xi_x;
        const double n_corr_y = nyf - xi_y;
        const double n_corr_z = nzf - xi_z;

        const double du_dx = 0.5 * (grad.dvx_dx(c0) + grad.dvx_dx(c1));
        const double du_dy = 0.5 * (grad.dvx_dy(c0) + grad.dvx_dy(c1));
        const double du_dz = 0.5 * (grad.dvx_dz(c0) + grad.dvx_dz(c1));

        const double dv_dx = 0.5 * (grad.dvy_dx(c0) + grad.dvy_dx(c1));
        const double dv_dy = 0.5 * (grad.dvy_dy(c0) + grad.dvy_dy(c1));
        const double dv_dz = 0.5 * (grad.dvy_dz(c0) + grad.dvy_dz(c1));

        const double dw_dx = 0.5 * (grad.dvz_dx(c0) + grad.dvz_dx(c1));
        const double dw_dy = 0.5 * (grad.dvz_dy(c0) + grad.dvz_dy(c1));
        const double dw_dz = 0.5 * (grad.dvz_dz(c0) + grad.dvz_dz(c1));

        const double dT_dx = 0.5 * (grad.dtmp_dx(c0) + grad.dtmp_dx(c1));
        const double dT_dy = 0.5 * (grad.dtmp_dy(c0) + grad.dtmp_dy(c1));
        const double dT_dz = 0.5 * (grad.dtmp_dz(c0) + grad.dtmp_dz(c1));

        const double div_v = du_dx + dv_dy + dw_dz;
        const double two_thirds_div_v = (2.0 / 3.0) * div_v;

        const double du_dn = (uR - uL) * inv_d + (du_dx * n_corr_x + du_dy * n_corr_y + du_dz * n_corr_z);
        const double dv_dn = (vR - vL) * inv_d + (dv_dx * n_corr_x + dv_dy * n_corr_y + dv_dz * n_corr_z);
        const double dw_dn = (wR - wL) * inv_d + (dw_dx * n_corr_x + dw_dy * n_corr_y + dw_dz * n_corr_z);
        const double dT_dn = (TR - TL) * inv_d + (dT_dx * n_corr_x + dT_dy * n_corr_y + dT_dz * n_corr_z);

        const double tau_nx = mu_eff * (du_dn + (du_dx * nxf + dv_dx * nyf + dw_dx * nzf) - two_thirds_div_v * nxf);
        const double tau_ny = mu_eff * (dv_dn + (du_dy * nxf + dv_dy * nyf + dw_dy * nzf) - two_thirds_div_v * nyf);
        const double tau_nz = mu_eff * (dw_dn + (du_dz * nxf + dv_dz * nyf + dw_dz * nzf) - two_thirds_div_v * nzf);

        const double qn = -k_eff * dT_dn;

        const double rho_f = eos_.density_Tp(T_f, p_f);
        const double nu_eff = mu_eff / rho_f;

        return ViscousFaceResult{
            .Fv1 = -tau_nx * Af,
            .Fv2 = -tau_ny * Af,
            .Fv3 = -tau_nz * Af,
            .Fv4 = (-(u_f * tau_nx + v_f * tau_ny + w_f * tau_nz) + qn) * Af,
            .lam_visc = (4.0 / 3.0) * nu_eff * inv_d * Af
        };
    }
    
    /**
     * @brief Evaluates combined (inviscid + viscous) fluxes over all interior and boundary faces in a single pass.
     */
    void compute_fluxes(fields::ConstPrimitiveView q,
                        fields::ConstPrimitiveGradView grad,
                        fields::ConstPrimitiveView phi,
                        const typename Recon::Geometry& geom,
                        fields::ResidualView<double> res,
                        double* CFD_RESTRICT lam,
                        const double* CFD_RESTRICT mut = nullptr,
                        double* CFD_RESTRICT mdot = nullptr) const noexcept {
        const std::size_t n_inner = static_cast<std::size_t>(mp_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mp_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mp_.n_cells);

        const LocalIndex* CFD_RESTRICT owner = mp_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mp_.face_neigh.data();
        const double* CFD_RESTRICT nx = mp_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mp_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mp_.face_normal_z.data();
        const double* CFD_RESTRICT area = mp_.face_area.data();

        const recon::ReconField rf{q, grad, phi};

        // 1. Interior faces sweep [0, n_inner)
        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[Phys::kNumVars];
            double qR[Phys::kNumVars];
            Recon::face_states(rf, geom, f, c0, c1, qL, qR);

            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[Phys::kNumVars];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            if constexpr (Phys::kNeedsFaceMdot) {
                mdot[f] = F[0];
            }

            double lam_f = Af * smax;

            if constexpr (Phys::kHasViscous) {
                const auto visc = compute_face_viscous_flux(f, c0, c1, q, grad, mut, nxf, nyf, nzf, Af);
                F[1] += visc.Fv1;
                F[2] += visc.Fv2;
                F[3] += visc.Fv3;
                F[4] += visc.Fv4;
                lam_f += visc.lam_visc;
            }

            // Flux leaves owner, enters neighbor
            res.res1[c0] += F[0];
            res.res2[c0] += F[1];
            res.res3[c0] += F[2];
            res.res4[c0] += F[3];
            res.res5[c0] += F[4];

            res.res1[c1] -= F[0];
            res.res2[c1] -= F[1];
            res.res3[c1] -= F[2];
            res.res4[c1] -= F[3];
            res.res5[c1] -= F[4];

            lam[c0] += lam_f;
            lam[c1] += lam_f;
        }

        // 2. Boundary faces sweep [n_inner, n_faces): owner vs. BC ghost
        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[Phys::kNumVars];
            double qR[Phys::kNumVars];
            Recon::boundary_face_states(rf, geom, f, c0, cg, qL, qR);

            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[Phys::kNumVars];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            if constexpr (Phys::kNeedsFaceMdot) {
                mdot[f] = F[0];
            }

            double lam_f = Af * smax;

            if constexpr (Phys::kHasViscous) {
                const auto visc = compute_face_viscous_flux(f, c0, cg, q, grad, mut, nxf, nyf, nzf, Af);
                F[1] += visc.Fv1;
                F[2] += visc.Fv2;
                F[3] += visc.Fv3;
                F[4] += visc.Fv4;
                lam_f += visc.lam_visc;
            }

            // Boundary face updates owner cell only
            res.res1[c0] += F[0];
            res.res2[c0] += F[1];
            res.res3[c0] += F[2];
            res.res4[c0] += F[3];
            res.res5[c0] += F[4];

            lam[c0] += lam_f;
        }
    }

    const mesh::MeshPart& mp_;
    EOS eos_;
    Phys phys_;
    typename Phys::Geometry phys_geom_;
};

} // namespace cfd::solver

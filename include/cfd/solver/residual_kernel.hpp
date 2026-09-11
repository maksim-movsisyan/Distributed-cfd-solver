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
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/solver/fluxes/viscous.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/fields/fields_view.hpp"
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
    ResidualKernel(const mesh::MeshPart& mesh, 
                   const mesh::MeshAuxConnectivity& aux_conn,
                   const mesh::MeshAuxGeometry& aux_geom,
                   const EOS eos, const Phys& phys)
        : mesh_(mesh), aux_conn_(aux_conn), aux_geom_(aux_geom), eos_(eos), phys_(phys) {}

    /**
     * @brief Evaluates the spatial residual and spectral radius across the local partition.
     *
     * @param[in]  q     Primitive cell states [p, u, v, w, T], halo- and BC-ghost-complete.
     * @param[in]  grad  Cell gradients (valid if kNeedsGradients == true).
     * @param[in]  phi   Cell gradient limiters in [0, 1].
     * @param[out] res   Accumulated flux balance vector.
     * @param[out] lam   Accumulated per-cell spectral radius (size >= n_cells).
     * @param[in]  mut   Eddy viscosity per cell (nullptr without turbulence).
     * @param[out] mdot  Face mass flux storage (nullptr unless physics modules request it).
     */
    void apply(fields::ConstPrimitiveView q,
               fields::ConstPrimitiveGradView grad,
               fields::ConstPrimitiveView phi,
               fields::ResidualView<double> res,
               double* CFD_RESTRICT lam,
               const double* CFD_RESTRICT mut = nullptr,
               double* CFD_RESTRICT mdot = nullptr) const noexcept {
        const std::size_t n_inner  = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mesh_.n_cells);
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
        compute_fluxes(q, grad, phi, res, lam, mut, mdot);
    }

    /**
     * @brief Computes rank-local net mass and total energy flux through each boundary patch.
     */
    void boundary_integrals(fields::ConstPrimitiveView q,
                            fields::ConstPrimitiveGradView grad,
                            fields::ConstPrimitiveView phi,
                            std::vector<double>& mass,
                            std::vector<double>& energy,
                            const double* CFD_RESTRICT mut = nullptr) const noexcept {
        const std::size_t n_inner  = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mesh_.n_cells);
        const std::size_t n_bfaces = n_faces - n_inner;
        const std::size_t n_total  = n_cells + n_bfaces;

        const std::size_t np      = mesh_.patches.size();

        mass.assign(np, 0.0);
        energy.assign(np, 0.0);

        const LocalIndex* CFD_RESTRICT owner = mesh_.face_owner.data();
        const mesh::PatchId* CFD_RESTRICT patch = mesh_.face_patch.data();
        const double* CFD_RESTRICT nx = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area = mesh_.face_area.data();

        const recon::ReconField rf{q, grad, phi};

        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[Phys::kNumVars];
            double qR[Phys::kNumVars];
            Recon::boundary_face_states(rf, mesh_, aux_geom_, f, c0, cg, qL, qR);

            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[Phys::kNumVars];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            double energy_flux = F[4];

            if constexpr (Phys::kHasViscous) {
                double F_visc[Phys::kNumVars];
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[cg] : 0.0;

                const double qL_visc[5] = { q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
                const double qR_visc[5] = { q.prs[cg], q.vx[cg], q.vy[cg], q.vz[cg], q.tmp[cg] };
                fluxes::ViscousFlux::face_flux(eos_, aux_geom_, f, c0, cg, n_total, qL_visc, qR_visc, 
                                               grad.vx_grad, grad.vy_grad, grad.vz_grad, grad.tmp_grad,
                                               mutL, mutR, nxf, nyf, nzf, Af, F_visc, lam_visc,
                                               phys_.prandtl(), phys_.prandtl_turb());
                energy_flux += F_visc[4];
            }

            const std::size_t p = static_cast<std::size_t>(patch[f]);
            mass[p]   += F[0];
            energy[p] += energy_flux;
        }
    }

private:    
    /**
     * @brief Evaluates combined (inviscid + viscous) fluxes over all interior and boundary faces in a single pass.
     */
    void compute_fluxes(fields::ConstPrimitiveView q,
                        fields::ConstPrimitiveGradView grad,
                        fields::ConstPrimitiveView phi,
                        fields::ResidualView<double> res,
                        double* CFD_RESTRICT lam,
                        const double* CFD_RESTRICT mut = nullptr,
                        double* CFD_RESTRICT mdot = nullptr) const noexcept {
        const std::size_t n_inner  = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mesh_.n_cells);
        const std::size_t n_bfaces = n_faces - n_inner;
        const std::size_t n_total  = n_cells + n_bfaces;

        const LocalIndex* CFD_RESTRICT owner = mesh_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mesh_.face_neigh.data();

        const double* CFD_RESTRICT nx = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area = mesh_.face_area.data();

        const recon::ReconField rf{q, grad, phi};

        // 1. Interior faces sweep [0, n_inner)
        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[Phys::kNumVars];
            double qR[Phys::kNumVars];
            Recon::face_states(rf, mesh_, aux_geom_, f, c0, c1, qL, qR);

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
                double F_visc[Phys::kNumVars];
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                const double qL_visc[5] = { q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
                const double qR_visc[5] = { q.prs[c1], q.vx[c1], q.vy[c1], q.vz[c1], q.tmp[c1] };
                fluxes::ViscousFlux::face_flux(eos_, aux_geom_, f, c0, c1, n_total, qL_visc, qR_visc, 
                                               grad.vx_grad, grad.vy_grad, grad.vz_grad, grad.tmp_grad,
                                               mutL, mutR, nxf, nyf, nzf, Af, F_visc, lam_visc,
                                               phys_.prandtl(), phys_.prandtl_turb());

                F[1] += F_visc[1];
                F[2] += F_visc[2];
                F[3] += F_visc[3];
                F[4] += F_visc[4];
                lam_f += lam_visc;
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
            Recon::boundary_face_states(rf, mesh_, aux_geom_, f, c0, cg, qL, qR);

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
                double F_visc[Phys::kNumVars];
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[cg] : 0.0;

                const double qL_visc[5] = { q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
                const double qR_visc[5] = { q.prs[cg], q.vx[cg], q.vy[cg], q.vz[cg], q.tmp[cg] };
                fluxes::ViscousFlux::face_flux(eos_, aux_geom_, f, c0, cg, n_total, qL_visc, qR_visc, 
                                               grad.vx_grad, grad.vy_grad, grad.vz_grad, grad.tmp_grad,
                                               mutL, mutR, nxf, nyf, nzf, Af, F_visc, lam_visc,
                                               phys_.prandtl(), phys_.prandtl_turb());

                F[1] += F_visc[1];
                F[2] += F_visc[2];
                F[3] += F_visc[3];
                F[4] += F_visc[4];
                lam_f += lam_visc;
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

    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    const mesh::MeshAuxGeometry& aux_geom_;
    EOS eos_;
    Phys phys_;
};

} // namespace cfd::solver

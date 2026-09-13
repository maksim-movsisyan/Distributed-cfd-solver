#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"
#include "cfd/solver/compressible/fluxes/viscous.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"
#include "cfd/solver/compressible/eos/state_conversions.hpp"
#include "cfd/solver/compressible/physics/physics_concepts.hpp"

namespace cfd::solver::compressible {

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
template <eos::EquationOfStatePolicy EOS, typename Flux, 
          typename Recon, physics::PhysicsGeneral Phys> requires numerics::recon::ReconstructionPolicy<Recon, Phys::kNumVars>
class ResidualKernel {
public:
    static constexpr std::size_t NVars = Phys::kNumVars;
    static_assert(NVars == 5, "Compressible mean-flow ResidualKernel requires exactly 5 variables");

    ResidualKernel(const mesh::MeshPart& mesh, 
                   const mesh::MeshAuxConnectivity& aux_conn,
                   const mesh::MeshAuxGeometry& aux_geom,
                   const EOS& eos, 
                   const Phys& phys)
        : mesh_(mesh), aux_conn_(aux_conn), aux_geom_(aux_geom), eos_(eos), phys_(phys) {}
        
    /**
     * @brief Computes spatial residuals and spectral radius for the entire local mesh.
     * 
     * @param[in]  q       Span of NVars pointers to cell primitive variables (SoA).
     * @param[in]  grad_x   Span of NVars pointers to d/dx gradient components.
     * @param[in]  grad_y   Span of NVars pointers to d/dy gradient components.
     * @param[in]  grad_z   Span of NVars pointers to d/dz gradient components.
     * @param[in]  phi      Span of NVars pointers to limiters.
     * @param[out] res      Span of NVars pointers to residual accumulators.
     * @param[out] lam      Cell stability spectral radius.
     * @param[in]  mut      Turbulent viscosity (nullptr if laminar/Euler flow).
     * @param[out] mdot     Face mass flow rates (nullptr if not required).
     */
    void apply(std::span<const double* const> q,
               std::span<const double* const> grad_x,
               std::span<const double* const> grad_y,
               std::span<const double* const> grad_z,
               std::span<const double* const> phi,
               std::span<double* const> res,
               double* CFD_RESTRICT lam,
               const double* CFD_RESTRICT mut = nullptr,
               double* CFD_RESTRICT mdot = nullptr) const noexcept {
        assert(q.size() == NVars);
        assert(grad_x.size() == NVars && grad_y.size() == NVars && grad_z.size() == NVars);
        assert(phi.size() == NVars);
        assert(res.size() == NVars);

        const std::size_t n_inner = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mesh_.n_cells);
        const std::size_t n_total = n_cells + (n_faces - n_inner);

        // 1. Reset residuals and spectral radii
        for (std::size_t v = 0; v < NVars; ++v) {
            std::fill_n(res[v], n_total, 0.0);
        }
        std::fill_n(lam, n_cells, 0.0);

        // 2. Fused flux evaluation (Euler Riemann solver + Navier-Stokes diffusion)
        compute_fluxes(q, grad_x, grad_y, grad_z, phi, res, lam, mut, mdot);
    }

    /**
     * @brief Computes rank-local net mass and total energy flux through each boundary patch.
     */
    void boundary_integrals(std::span<const double* const> q,
                            std::span<const double* const> grad_x,
                            std::span<const double* const> grad_y,
                            std::span<const double* const> grad_z,
                            std::span<const double* const> phi,
                            std::vector<double>& mass,
                            std::vector<double>& energy,
                            const double* CFD_RESTRICT mut = nullptr) const noexcept {
        assert(q.size() == NVars);
        assert(grad_x.size() == NVars && grad_y.size() == NVars && grad_z.size() == NVars);
        assert(phi.size() == NVars);

        const std::size_t n_inner = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mesh_.n_cells);
        const std::size_t np      = mesh_.patches.size();

        mass.assign(np, 0.0);
        energy.assign(np, 0.0);

        const LocalIndex* CFD_RESTRICT owner    = mesh_.face_owner.data();
        const mesh::PatchId* CFD_RESTRICT patch = mesh_.face_patch.data();
        const double* CFD_RESTRICT nx           = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny           = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz           = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area         = mesh_.face_area.data();
        
        [[maybe_unused]] fluxes::ViscousViews vv{};
        if constexpr (Phys::kHasViscous) {
            vv = fluxes::ViscousViews::from_spans(q, grad_x, grad_y, grad_z);
        }
        
        const numerics::recon::ReconBatchField<NVars> rf(q.data(), grad_x.data(), grad_y.data(), grad_z.data(), phi.data());

        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[5];
            double qR[5];
            Recon::template boundary_face_states<NVars>(rf, mesh_, aux_geom_, f, c0, cg, qL, qR);

            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[5];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            double energy_flux = F[4];

            if constexpr (Phys::kHasViscous) {
                double F_visc[5]{0.0};
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[cg] : 0.0;

                fluxes::ViscousFlux::face_flux(
                    eos_, aux_geom_, f, c0, cg,
                    vv, mutL, mutR, nxf, nyf, nzf, Af, 
                    F_visc, lam_visc,
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
    void compute_fluxes(std::span<const double* const> q,
                        std::span<const double* const> grad_x,
                        std::span<const double* const> grad_y,
                        std::span<const double* const> grad_z,
                        std::span<const double* const> phi,
                        std::span<double* const> res,
                        double* CFD_RESTRICT lam,
                        const double* CFD_RESTRICT mut,
                        double* CFD_RESTRICT mdot) const noexcept {
        const std::size_t n_inner = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mesh_.n_cells);

        const LocalIndex* CFD_RESTRICT owner = mesh_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mesh_.face_neigh.data();

        const double* CFD_RESTRICT nx   = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny   = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz   = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area = mesh_.face_area.data();
    
        [[maybe_unused]] fluxes::ViscousViews vv{};
        if constexpr (Phys::kHasViscous) {
            vv = fluxes::ViscousViews::from_spans(q, grad_x, grad_y, grad_z);
        }
        
        const numerics::recon::ReconBatchField<NVars> rf(q.data(), grad_x.data(), grad_y.data(), grad_z.data(), phi.data());

        // 1. Interior faces sweep [0, n_inner)
        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[5];
            double qR[5];
            Recon::template face_states<NVars>(rf, mesh_, aux_geom_, f, c0, c1, qL, qR);

            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[5];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            if constexpr (Phys::kNeedsFaceMdot) {
                mdot[f] = F[0];
            }

            double lam_f = Af * smax;

            if constexpr (Phys::kHasViscous) {
                double F_visc[5]{0.0};
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                fluxes::ViscousFlux::face_flux(
                    eos_, aux_geom_, f, c0, c1,
                    vv, mutL, mutR, nxf, nyf, nzf, Af, 
                    F_visc, lam_visc,
                    phys_.prandtl(), phys_.prandtl_turb());

                for (std::size_t v = 1; v < 5; ++v) {
                    F[v] += F_visc[v];
                }
                lam_f += lam_visc;
            }

            // Flux leaves owner, enters neighbor
            for (std::size_t v = 0; v < 5; ++v) {
                res[v][c0] += F[v];
                res[v][c1] -= F[v];
            }

            lam[c0] += lam_f;
            lam[c1] += lam_f;
        }

        // 2. Boundary faces sweep [n_inner, n_faces): owner vs. BC ghost
        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            double qL[5];
            double qR[5];
            Recon::template boundary_face_states<NVars>(rf, mesh_, aux_geom_, f, c0, cg, qL, qR);

            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double F[5];
            double smax = 0.0;
            Flux::face_flux(eos_, UL, UR, nxf, nyf, nzf, Af, F, smax);

            if constexpr (Phys::kNeedsFaceMdot) {
                mdot[f] = F[0];
            }

            double lam_f = Af * smax;

            if constexpr (Phys::kHasViscous) {
                double F_visc[5]{0.0};
                double lam_visc = 0.0;
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[cg] : 0.0;

                fluxes::ViscousFlux::face_flux(
                    eos_, aux_geom_, f, c0, cg,
                    vv, mutL, mutR, nxf, nyf, nzf, Af, 
                    F_visc, lam_visc,
                    phys_.prandtl(), phys_.prandtl_turb());

                for (std::size_t v = 1; v < 5; ++v) {
                    F[v] += F_visc[v];
                }
                lam_f += lam_visc;
            }

            // Boundary face updates owner cell only
            for (std::size_t v = 0; v < 5; ++v) {
                res[v][c0] += F[v];
            }

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

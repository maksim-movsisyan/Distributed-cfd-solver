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
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"


namespace cfd::solver {

namespace {

inline std::size_t bsr_get_pos(const LocalIndex* CFD_RESTRICT row_ptr_,
                               const LocalIndex* CFD_RESTRICT cols_,
                               std::size_t row, std::size_t col) {
    const std::size_t ptr1 = static_cast<std::size_t>(row_ptr_[row]);
    const std::size_t ptr2 = static_cast<std::size_t>(row_ptr_[row + 1]);     
    
    const LocalIndex* row_start = cols_ + ptr1;
    const LocalIndex* row_end   = cols_ + ptr2;

    const LocalIndex* it = std::lower_bound(row_start, row_end, static_cast<LocalIndex>(col));
 
    assert(it != row_end && *it == col && "CFD Error: Element missing in dual graph!");

    return static_cast<std::size_t>(it - cols_);
}

}

/**
 * @class JacobianKernel
 * @brief High-performance cell-centered finite volume spatial inexact jacobian evaluator.
 * 
 * @tparam EOS   Thermodynamic Equation of State conforming to eos::EquationOfState
 * @tparam Flux  Numerical flux policy (e.g., riemann::HllcFlux)
 * @tparam Phys  Flow equation set
 */
template <eos::EquationOfState EOS, typename Flux, physics::PhysicsGeneral Phys>
class JacobianKernel {
public:
    JacobianKernel(const mesh::MeshPart& mesh, 
                   const mesh::MeshAuxConnectivity& aux_conn,
                   const mesh::MeshAuxGeometry& aux_geom,
                   const EOS eos, const Phys& phys)
        : mesh_(mesh), aux_conn_(aux_conn), aux_geom_(aux_geom), eos_(eos), phys_(phys) {}

    /**
     * @brief Evaluates the spatial residual and spectral radius across the local partition.
     *
     * @param[in]  q       Primitive cell states [p, u, v, w, T], halo- and BC-ghost-complete.
     * @param[out] values, row_ptr, cols, diag_idx  BSR Matrix 
     * @param[in]  alpha   Cell volume devided by time step (may be summ of time + pseudo-time)
     * @param[in]  mut     Eddy viscosity per cell (nullptr without turbulence).
     */
    void apply(fields::ConstPrimitiveView q,
               const LocalIndex* CFD_RESTRICT row_ptr,
               const LocalIndex* CFD_RESTRICT cols,
               const LocalIndex* CFD_RESTRICT diag_idx,
               double* CFD_RESTRICT values,
               const double* CFD_RESTRICT alpha,
               const double* CFD_RESTRICT mut = nullptr) const noexcept {
        const std::size_t bs = Phys::kNumVars;
        const std::size_t bs2 = bs * bs;
        // 1. Reset values
        const std::size_t n_own  = static_cast<std::size_t>(mesh_.n_own);
        std::fill(values, values + row_ptr[n_own] * Phys::kNumVars * Phys::kNumVars, 0.0);

        // 2. Fused flux evaluation (Euler Riemann solver + Navier-Stokes diffusion)
        compute_jacobians(q, row_ptr, cols, diag_idx, values, mut);

        // 3. Add diagonal contribution
        if (alpha != nullptr) {
            for (std::size_t c = 0; c < n_own; ++c) {
                const std::size_t pos = static_cast<std::size_t>(diag_idx[c]) * bs2;
                const double diag_val = alpha[c];

                for (std::size_t i = 0; i < bs; ++i) {
                    values[pos + i * (bs + 1)] += diag_val;
                }
            }
        }
    }

private:    
    /**
     * @brief Evaluates combined (inviscid + viscous) jacobians over all interior and boundary faces in a single pass.
     */
    void compute_jacobians(fields::ConstPrimitiveView q,
                           const LocalIndex* CFD_RESTRICT row_ptr,
                           const LocalIndex* CFD_RESTRICT cols,
                           const LocalIndex* CFD_RESTRICT diag_idx,
                           double* CFD_RESTRICT values,
                           const double* CFD_RESTRICT mut = nullptr) const noexcept {
        const std::size_t n_inter  = static_cast<std::size_t>(mesh_.n_internal_faces);
        const std::size_t n_inner  = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mesh_.n_cells);

        const std::size_t bs = Phys::kNumVars;
        const std::size_t bs2 = bs * bs;
        
        const LocalIndex* CFD_RESTRICT owner = mesh_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mesh_.face_neigh.data();

        const double* CFD_RESTRICT nx = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area = mesh_.face_area.data();

        // 1. Internal faces sweep [0, n_inter) => all cells has a row in matrix
        for (std::size_t f = 0; f < n_inter; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[Phys::kNumVars] = {q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
            const double qR[Phys::kNumVars] = {q.prs[c1], q.vx[c1], q.vy[c1], q.vz[c1], q.tmp[c1] };
            
            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[bs2];
            double dFR[bs2];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[bs2];
                double dFR_visc[bs2];
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af, dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < bs2; ++i) {
                    dFL[i] += dFL_visc[i];
                    dFR[i] += dFR_visc[i];
                } 
            }


            std::size_t pos;
            // owner row, owner col (diagonal)
            pos = static_cast<std::size_t>(diag_idx[c0]) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] += dFL[i];
            // owner row, neighbor col
            pos = bsr_get_pos(row_ptr, cols, c0, c1) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] += dFR[i];

            // neighbor row, neighbor col (diagonal)
            pos = static_cast<std::size_t>(diag_idx[c1]) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] -= dFR[i];
            // owner row, neighbor col
            pos = bsr_get_pos(row_ptr, cols, c1, c0) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] -= dFL[i];

        }

        // 2. Inner faces sweep [n_inter, n_inner) => only owner cells has a row, neigbhor (ghost) cell has a col
        for (std::size_t f = n_inter; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[Phys::kNumVars] = {q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
            const double qR[Phys::kNumVars] = {q.prs[c1], q.vx[c1], q.vy[c1], q.vz[c1], q.tmp[c1] };
            
            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[bs2];
            double dFR[bs2];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[bs2];
                double dFR_visc[bs2];
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af, dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < bs2; ++i) {
                    dFL[i] += dFL_visc[i];
                    dFR[i] += dFR_visc[i];
                } 
            }


            std::size_t pos;
            // owner row, owner col (diagonal)
            pos = static_cast<std::size_t>(diag_idx[c0]) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] += dFL[i];
            // owner row, neighbor col
            pos = bsr_get_pos(row_ptr, cols, c0, c1) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] += dFR[i];
        }

        // 3. Boundary faces sweep [n_inner, n_inner) => only owner cells has a row (inexact, can lead to negative pressure/temperature, tobeimproved)
        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[Phys::kNumVars] = {q.prs[c0], q.vx[c0], q.vy[c0], q.vz[c0], q.tmp[c0] };
            const double qR[Phys::kNumVars] = {q.prs[cg], q.vx[cg], q.vy[cg], q.vz[cg], q.tmp[cg] };

            double UL[Phys::kNumVars];
            double UR[Phys::kNumVars];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[bs2];
            double dFR[bs2];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[bs2];
                double dFR_visc[bs2];
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[cg] : 0.0;

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af, dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < bs2; ++i) {
                    dFL[i] += dFL_visc[i];
                    dFR[i] += dFR_visc[i];
                } 
            }


            std::size_t pos;
            // owner row, owner col (diagonal)
            pos = static_cast<std::size_t>(diag_idx[c0]) * bs2;
            for (std::size_t i = 0; i < bs2; ++i) values[pos + i] += dFL[i]; 
        }
    }

    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    const mesh::MeshAuxGeometry& aux_geom_;
    EOS eos_;
    Phys phys_;
};

} // namespace cfd::solver

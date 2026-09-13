#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <span>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/solver/compressible/fluxes/viscous.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"
#include "cfd/solver/compressible/eos/state_conversions.hpp"
#include "cfd/solver/compressible/physics/physics_concepts.hpp"


namespace cfd::solver::compressible {

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
 * @tparam EOS   Thermodynamic Equation of State conforming to eos::EquationOfStatePolicy
 * @tparam Flux  Numerical flux policy (e.g., riemann::HllcFlux)
 * @tparam Phys  Flow equation set
 */
template <eos::EquationOfStatePolicy EOS, typename Flux, physics::PhysicsGeneral Phys>
class JacobianKernel {
public:
    static constexpr std::size_t NVars = Phys::kNumVars;
    static_assert(NVars == 5, "Compressible mean-flow JacobianKernel currently requires exactly 5 variables");
    
    static constexpr std::size_t BlockSize = NVars * NVars; // 25 entries per block

    JacobianKernel(const mesh::MeshPart& mesh, 
                   const mesh::MeshAuxConnectivity& aux_conn,
                   const mesh::MeshAuxGeometry& aux_geom,
                   const EOS& eos, 
                   const Phys& phys)
        : mesh_(mesh), aux_conn_(aux_conn), aux_geom_(aux_geom), eos_(eos), phys_(phys) {}

    /**
     * @brief Fills the BSR matrix blocks of the Jacobian and adds the diagonal contribution (V / dt).
     *
     * @param[in]  q         Span of NVars pointers to cell primitives in SoA layout.
     * @param[in]  row_ptr   Pointers to the beginning of rows in the BSR matrix.
     * @param[in]  cols      Column indices of the BSR matrix.
     * @param[in]  diag_idx  Direct indices of diagonal blocks for each cell.
     * @param[out] values    Values array of the BSR matrix blocks (size: row_ptr[n_own] * BlockSize).
     * @param[in]  alpha     Cell diagonal contribution (V / dt), nullptr if not required.
     * @param[in]  mut       Turbulent viscosity (nullptr for laminar/Euler computation).
     */
    void apply(std::span<const double* const> q,
               const LocalIndex* CFD_RESTRICT row_ptr,
               const LocalIndex* CFD_RESTRICT cols,
               const LocalIndex* CFD_RESTRICT diag_idx,
               double* CFD_RESTRICT values,
               const double* CFD_RESTRICT alpha,
               const double* CFD_RESTRICT mut = nullptr) const noexcept {
        assert(q.size() == NVars && "Mismatch in number of primitive fields");

        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);
        const std::size_t nnz_blocks = static_cast<std::size_t>(row_ptr[n_own]);

        // 1. Reset values
        std::fill_n(values, nnz_blocks * BlockSize, 0.0);

        // 2. Fused flux evaluation (Euler Riemann solver + Navier-Stokes diffusion)
        compute_jacobians(q, row_ptr, cols, diag_idx, values, mut);

        // 3. Add diagonal contribution
        if (alpha != nullptr) {
            for (std::size_t c = 0; c < n_own; ++c) {
                const std::size_t pos = static_cast<std::size_t>(diag_idx[c]) * BlockSize;
                const double diag_val = alpha[c];

                for (std::size_t i = 0; i < NVars; ++i) {
                    values[pos + i * (NVars + 1)] += diag_val;
                }
            }
        }
    }

private:    
    /**
     * @brief Evaluates combined (inviscid + viscous) jacobians over all interior and boundary faces in a single pass.
     */
    void compute_jacobians(std::span<const double* const> q,
                           const LocalIndex* CFD_RESTRICT row_ptr,
                           const LocalIndex* CFD_RESTRICT cols,
                           const LocalIndex* CFD_RESTRICT diag_idx,
                           double* CFD_RESTRICT values,
                           const double* CFD_RESTRICT mut = nullptr) const noexcept {
        const std::size_t n_inter = static_cast<std::size_t>(mesh_.n_internal_faces);
        const std::size_t n_inner = static_cast<std::size_t>(mesh_.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mesh_.n_faces);
        const std::size_t n_cells = static_cast<std::size_t>(mesh_.n_cells);

        const LocalIndex* CFD_RESTRICT owner = mesh_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mesh_.face_neigh.data();

        const double* CFD_RESTRICT nx   = mesh_.face_normal_x.data();
        const double* CFD_RESTRICT ny   = mesh_.face_normal_y.data();
        const double* CFD_RESTRICT nz   = mesh_.face_normal_z.data();
        const double* CFD_RESTRICT area = mesh_.face_area.data();

        // Cache primitive field pointers
        const double* CFD_RESTRICT prs = q[0];
        const double* CFD_RESTRICT vx  = q[1];
        const double* CFD_RESTRICT vy  = q[2];
        const double* CFD_RESTRICT vz  = q[3];
        const double* CFD_RESTRICT tmp = q[4];

        // 1. Internal faces sweep [0, n_inter): Both owner and neighbor cells have rows in matrix
        for (std::size_t f = 0; f < n_inter; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[5] = {prs[c0], vx[c0], vy[c0], vz[c0], tmp[c0]};
            const double qR[5] = {prs[c1], vx[c1], vy[c1], vz[c1], tmp[c1]};
            
            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[BlockSize];
            double dFR[BlockSize];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[BlockSize];
                double dFR_visc[BlockSize];
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af,
                                                        dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < BlockSize; ++i) {
                    dFL[i] += dFL_visc[i];
                    dFR[i] += dFR_visc[i];
                } 
            }

            // Owner cell row: A(c0, c0) += dFL, A(c0, c1) += dFR
            const std::size_t pos_c0_c0 = static_cast<std::size_t>(diag_idx[c0]) * BlockSize;
            const std::size_t pos_c0_c1 = bsr_get_pos(row_ptr, cols, c0, c1) * BlockSize;

            for (std::size_t i = 0; i < BlockSize; ++i) {
                values[pos_c0_c0 + i] += dFL[i];
                values[pos_c0_c1 + i] += dFR[i];
            }

            // Neighbor cell row: A(c1, c1) -= dFR, A(c1, c0) -= dFL
            const std::size_t pos_c1_c1 = static_cast<std::size_t>(diag_idx[c1]) * BlockSize;
            const std::size_t pos_c1_c0 = bsr_get_pos(row_ptr, cols, c1, c0) * BlockSize;

            for (std::size_t i = 0; i < BlockSize; ++i) {
                values[pos_c1_c1 + i] -= dFR[i];
                values[pos_c1_c0 + i] -= dFL[i];
            }
        }

        // 2. Inter-rank faces sweep [n_inter, n_inner): Owner has a row, neighbor (MPI ghost) only has a column
        for (std::size_t f = n_inter; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[5] = {prs[c0], vx[c0], vy[c0], vz[c0], tmp[c0]};
            const double qR[5] = {prs[c1], vx[c1], vy[c1], vz[c1], tmp[c1]};
            
            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[BlockSize];
            double dFR[BlockSize];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[BlockSize];
                double dFR_visc[BlockSize];
                const double mutL = mut ? mut[c0] : 0.0;
                const double mutR = mut ? mut[c1] : 0.0;

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af, 
                                                        dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < BlockSize; ++i) {
                    dFL[i] += dFL_visc[i];
                    dFR[i] += dFR_visc[i];
                } 
            }

            const std::size_t pos_c0_c0 = static_cast<std::size_t>(diag_idx[c0]) * BlockSize;
            const std::size_t pos_c0_c1 = bsr_get_pos(row_ptr, cols, c0, c1) * BlockSize;

            for (std::size_t i = 0; i < BlockSize; ++i) {
                values[pos_c0_c0 + i] += dFL[i];
                values[pos_c0_c1 + i] += dFR[i];
            }
        }

        // 3. Boundary faces sweep [n_inner, n_faces): Only owner cells have rows (inexact Picard boundary Jacobian)
        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);
            const double nxf = nx[f], nyf = ny[f], nzf = nz[f], Af = area[f];

            const double qL[5] = {prs[c0], vx[c0], vy[c0], vz[c0], tmp[c0]};
            const double qR[5] = {prs[cg], vx[cg], vy[cg], vz[cg], tmp[cg]};

            double UL[5];
            double UR[5];
            eos::primitives_pT_to_conserved(eos_, qL[0], qL[1], qL[2], qL[3], qL[4], UL);
            eos::primitives_pT_to_conserved(eos_, qR[0], qR[1], qR[2], qR[3], qR[4], UR);

            double dFL[BlockSize];
            double dFR[BlockSize];
            Flux::face_flux_jacobian(eos_, UL, UR, nxf, nyf, nzf, Af, dFL, dFR);

            if constexpr (Phys::kHasViscous) {
                double dFL_visc[BlockSize];
                double dFR_visc[BlockSize];
                const double mutL = mut ? mut[c0] : 0.0;
                // Boundary ghost cells default to zero eddy viscosity if mut is not allocated for boundary ghost cells
                const double mutR = 0.0; 

                fluxes::ViscousFlux::face_flux_jacobian(eos_, aux_geom_, f, qL, qR, mutL, mutR, Af, 
                                                        dFL_visc, dFR_visc,
                                                        phys_.prandtl(), phys_.prandtl_turb());
                
                for (std::size_t i = 0; i < BlockSize; ++i) {
                    dFL[i] += dFL_visc[i];
                } 
            }

            const std::size_t pos_c0_c0 = static_cast<std::size_t>(diag_idx[c0]) * BlockSize;
            for (std::size_t i = 0; i < BlockSize; ++i) {
                values[pos_c0_c0 + i] += dFL[i]; 
            }
        }
    }

    const mesh::MeshPart& mesh_;
    const mesh::MeshAuxConnectivity& aux_conn_;
    const mesh::MeshAuxGeometry& aux_geom_;
    EOS eos_;
    Phys phys_;
};

} // namespace cfd::solver

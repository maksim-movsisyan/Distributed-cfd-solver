// Residual evaluation R(U): the reusable functional behind every time
// integrator (explicit stages now, matrix-free implicit later).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver {

/**
 * @class ResidualKernel
 * @brief High-performance cell-centered finite volume spatial residual evaluator.
 * 
 * Accumulates convective flux balance:
 *   Res = sum_{faces} (F_num * Area)
 * and cell spectral wave radii:
 *   Lambda = sum_{faces} (Area * s_max)
 *
 * Update rule in time integrator: U^{n+1} = U^n - (dt / V) * Res
 *
 * @tparam EOS  Thermodynamic Equation of State conforming to eos::EquationOfState
 * @tparam Flux Numerical flux policy (e.g., riemann::HllcFlux)
 */
template <eos::EquationOfState EOS, typename Flux>
class ResidualKernel {
public:
    ResidualKernel(const mesh::MeshPart& mp, const EOS eos)
        : mp_(mp), eos_(eos) {}

    /**
     * @brief Evaluates the spatial residual and spectral radius across the local partition.
     * 
     * @param[in]  u    Conservative variables [rho, rhou, rhov, rhow, rhoE] (fully populated with ghosts).
     * @param[out] res  Accumulated flux balance vector.
     * @param[out] lam  Accumulated per-cell spectral radius (size >= n_cells).
     */
    void apply(fields::ConservativeView<const double> u,
               fields::ResidualView<double> res,
               double* CFD_RESTRICT lam) const noexcept {
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

        // Mesh geometry pointers
        const LocalIndex* CFD_RESTRICT owner = mp_.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mp_.face_neigh.data();
        const double* CFD_RESTRICT nx = mp_.face_normal_x.data();
        const double* CFD_RESTRICT ny = mp_.face_normal_y.data();
        const double* CFD_RESTRICT nz = mp_.face_normal_z.data();
        const double* CFD_RESTRICT area = mp_.face_area.data();

        // 2. Interior faces sweep [0, n_inner)
        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);

            const double UL[eos::kNumVars] = {
                u.rho[c0], u.rhou[c0], u.rhov[c0], u.rhow[c0], u.rhoE[c0]
            };
            const double UR[eos::kNumVars] = {
                u.rho[c1], u.rhou[c1], u.rhov[c1], u.rhow[c1], u.rhoE[c1]
            };

            double F[eos::kNumVars];
            double smax = 0.0;

            Flux::face_flux(eos_, UL, UR, nx[f], ny[f], nz[f], area[f], F, smax);

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

            const double w = area[f] * smax;
            lam[c0] += w;
            lam[c1] += w;
        }

        // 3. Boundary faces sweep [n_inner, n_faces)
        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);

            const double UL[eos::kNumVars] = {
                u.rho[c0], u.rhou[c0], u.rhov[c0], u.rhow[c0], u.rhoE[c0]
            };
            const double UR[eos::kNumVars] = {
                u.rho[cg], u.rhou[cg], u.rhov[cg], u.rhow[cg], u.rhoE[cg]
            };

            double F[eos::kNumVars];
            double smax = 0.0;

            Flux::face_flux(eos_, UL, UR, nx[f], ny[f], nz[f], area[f], F, smax);

            // Boundary face updates owner cell only
            res.res1[c0] += F[0];
            res.res2[c0] += F[1];
            res.res3[c0] += F[2];
            res.res4[c0] += F[3];
            res.res5[c0] += F[4];

            lam[c0] += area[f] * smax;
        }
    }

    /**
     * @brief Computes rank-local net mass and total energy flux through each boundary patch.
     */
    void boundary_integrals(fields::ConservativeView<const double> u,
                            std::vector<double>& mass,
                            std::vector<double>& energy) const noexcept {
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

        for (std::size_t f = n_inner; f < n_faces; ++f) {
            const auto c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells + (f - n_inner);

            const double UL[eos::kNumVars] = {
                u.rho[c0], u.rhou[c0], u.rhov[c0], u.rhow[c0], u.rhoE[c0]
            };
            const double UR[eos::kNumVars] = {
                u.rho[cg], u.rhou[cg], u.rhov[cg], u.rhow[cg], u.rhoE[cg]
            };

            double F[eos::kNumVars];
            double smax = 0.0;

            Flux::face_flux(eos_, UL, UR, nx[f], ny[f], nz[f], area[f], F, smax);

            const std::size_t p = static_cast<std::size_t>(patch[f]);
            mass[p]   += F[0];
            energy[p] += F[4];
        }
    }

    [[nodiscard]] const EOS& eos() const noexcept { return eos_; }

private:
    const mesh::MeshPart& mp_;
    EOS eos_;
};

} // namespace cfd::solver
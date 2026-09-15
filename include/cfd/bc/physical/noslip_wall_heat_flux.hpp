#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/numerics/boundary/boundary_gradients.hpp"
#include "cfd/numerics/boundary/boundary_traversal.hpp"
#include "cfd/numerics/boundary/boundary_values.hpp"
#include "cfd/bc/physical/physical_bc.hpp"

namespace cfd::bc::physical {

/** 
 * @brief Parameters for No-Slip Wall with specified heat flux (temperature gradient).
 * 
 * @note tmp_grad_wall represents the normal temperature gradient dT/dn [K/m].
 *       For an adiabatic wall (q_w = 0), set tmp_grad_wall = 0.0.
 *       For a specified heat flux q_w [W/m^2], dT/dn = -q_w / lambda.
 */
struct NoSlipWallHeatFluxParams {
    double vx_wall{0.0};       ///< Wall velocity X [m/s]
    double vy_wall{0.0};       ///< Wall velocity Y [m/s]
    double vz_wall{0.0};       ///< Wall velocity Z [m/s]
    double tmp_grad_wall{0.0}; ///< Normal temperature gradient dT/dn [K/m]

    static constexpr NoSlipWallHeatFluxParams adiabatic() noexcept {
        return NoSlipWallHeatFluxParams{0.0, 0.0, 0.0, 0.0};
    }

    static constexpr NoSlipWallHeatFluxParams fixed_gradient(const double grad_T) noexcept {
        return NoSlipWallHeatFluxParams{0.0, 0.0, 0.0, grad_T};
    }

    static constexpr NoSlipWallHeatFluxParams moving_gradient(const double u,
                                                              const double v,
                                                              const double w,
                                                              const double grad_T) noexcept {
        return NoSlipWallHeatFluxParams{u, v, w, grad_T};
    }
};

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for No-Slip Wall with specified heat flux.
 * For incompressible flows (NVars == 4), normal/metric arrays are not fetched from memory.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void no_slip_wall_heat_flux_kernel(std::span<double* const> q,
                                          const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend,
                                          const NoSlipWallHeatFluxParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache primitive field pointers with restrict in registers
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    const double vx_w          = p.vx_wall;
    const double vy_w          = p.vy_wall;
    const double vz_w          = p.vz_wall;
    const double tmp_grad_wall = p.tmp_grad_wall;

    if constexpr (NVars == 4) {
        // Incompressible path: pure Dirichlet velocities + extrapolated pressure (no metrics loaded)
        numerics::boundary::for_each_boundary_face(m, fbeg, fend,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
                numerics::boundary::apply_extrapolate_value(lq[0][gh], lq[0][in]);
                numerics::boundary::apply_dirichlet_value(lq[1][gh], lq[1][in], vx_w);
                numerics::boundary::apply_dirichlet_value(lq[2][gh], lq[2][in], vy_w);
                numerics::boundary::apply_dirichlet_value(lq[3][gh], lq[3][in], vz_w);
            });
    } else {
        // Compressible path: requires normal projection distance for Neumann temperature condition
        numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                double /*nx*/, double /*ny*/, double /*nz*/, double rcfn_inv) noexcept {
                // 1. Extrapolate pressure: dp/dn = 0
                numerics::boundary::apply_extrapolate_value(lq[0][gh], lq[0][in]);

                // 2. Dirichlet velocities: v_ghost = 2 * v_wall - v_in
                numerics::boundary::apply_dirichlet_value(lq[1][gh], lq[1][in], vx_w);
                numerics::boundary::apply_dirichlet_value(lq[2][gh], lq[2][in], vy_w);
                numerics::boundary::apply_dirichlet_value(lq[3][gh], lq[3][in], vz_w);

                // 3. Neumann temperature condition: dT/dn = tmp_grad_wall
                const double rcfn = 1.0 / rcfn_inv;
                numerics::boundary::apply_neumann_value(lq[4][gh], lq[4][in], tmp_grad_wall, rcfn);
                lq[4][gh] = std::max(lq[4][gh], 1.0);
            });
    }
}

/** 
 * @brief Fills ghost cells with gradients for No-Slip Wall with specified heat flux.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void no_slip_wall_heat_flux_grad_kernel(std::span<const double* const> q,
                                               std::span<double* const> gx,
                                               std::span<double* const> gy,
                                               std::span<double* const> gz,
                                               const mesh::MeshPart& m,
                                               const LocalIndex fbeg,
                                               const LocalIndex fend,
                                               const NoSlipWallHeatFluxParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    const double* CFD_RESTRICT lq[NVars];
    double* CFD_RESTRICT lgx[NVars];
    double* CFD_RESTRICT lgy[NVars];
    double* CFD_RESTRICT lgz[NVars];

    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v]  = q[v];
        lgx[v] = gx[v];
        lgy[v] = gy[v];
        lgz[v] = gz[v];
    }

    const double v_wall[3]     = {p.vx_wall, p.vy_wall, p.vz_wall};
    const double tmp_grad_wall = p.tmp_grad_wall;

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // 1. Zero-order extrapolation for pressure gradient: dp/dn = 0
            numerics::boundary::apply_extrapolate_gradient(
                lgx[0][gh], lgy[0][gh], lgz[0][gh],
                lgx[0][in], lgy[0][in], lgz[0][in]);

            // 2. Velocity gradient: fixed Dirichlet value at wall
            for (std::size_t d = 0; d < 3; ++d) {
                const std::size_t v = 1 + d;
                numerics::boundary::apply_dirichlet_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in],
                    lq[v][in], v_wall[d],
                    nx, ny, nz, rcfn_inv);
            }

            // 3. Temperature gradient: fixed Neumann normal gradient (dT/dn = tmp_grad_wall)
            if constexpr (NVars >= 5) {
                numerics::boundary::apply_neumann_gradient(
                    lgx[4][gh], lgy[4][gh], lgz[4][gh],
                    lgx[4][in], lgy[4][in], lgz[4][in],
                    tmp_grad_wall, nx, ny, nz);
            }
        });
}

} // namespace kernels

/**
 * @class NoSlipWallHeatFluxBC
 * @brief No-Slip Wall boundary condition with specified normal temperature gradient / heat flux.
 * Automatically functions as a clean no-slip wall when executed in incompressible solvers.
 */
class NoSlipWallHeatFluxBC final : public BoundaryCondition {
public:
    NoSlipWallHeatFluxBC(std::string zone,
                         const LocalIndex fbeg,
                         const LocalIndex fend,
                         const NoSlipWallHeatFluxParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "NoSlipWallHeatFluxBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::no_slip_wall_heat_flux_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::no_slip_wall_heat_flux_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && gx.size() >= 4 && gy.size() >= 4 && gz.size() >= 4);
        assert(q.size() == gx.size() && gx.size() == gy.size() && gy.size() == gz.size());

        if (q.size() >= 5) {
            kernels::no_slip_wall_heat_flux_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::no_slip_wall_heat_flux_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void apply_momentum_bc(std::span<double*> diag_u,
                           std::span<double*> rhs,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs);
        static_cast<void>(diag_u);
    }

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWallHeatFlux; }

private:
    NoSlipWallHeatFluxParams m_p;
};

} // namespace cfd::bc::physical
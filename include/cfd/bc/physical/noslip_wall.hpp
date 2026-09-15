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
 * @brief Parameters for Isothermal No-Slip Wall Boundary Condition.
 * Direct physical parameters without EOS coupling.
 */
struct NoSlipWallParams {
    double vx_wall{0.0};     ///< Wall velocity X [m/s]
    double vy_wall{0.0};     ///< Wall velocity Y [m/s]
    double vz_wall{0.0};     ///< Wall velocity Z [m/s]
    double tmp_wall{288.15}; ///< Wall temperature [K]
};

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for Isothermal No-Slip Wall.
 * For incompressible flow (NVars == 4), temperature logic is completely eliminated at compile time.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void no_slip_wall_kernel(std::span<double* const> q,
                                const mesh::MeshPart& m,
                                const LocalIndex fbeg,
                                const LocalIndex fend,
                                const NoSlipWallParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache primitive field pointers with restrict in registers
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    const double vx_w  = p.vx_wall;
    const double vy_w  = p.vy_wall;
    const double vz_w  = p.vz_wall;
    const double tmp_w = p.tmp_wall;

    // Fast traversal: pure Dirichlet/extrapolation does not require face normals
    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            // 1. Zero normal pressure gradient: dp/dn = 0
            numerics::boundary::apply_extrapolate_value(lq[0][gh], lq[0][in]);

            // 2. Dirichlet velocity: v_ghost = 2 * v_wall - v_in
            numerics::boundary::apply_dirichlet_value(lq[1][gh], lq[1][in], vx_w);
            numerics::boundary::apply_dirichlet_value(lq[2][gh], lq[2][in], vy_w);
            numerics::boundary::apply_dirichlet_value(lq[3][gh], lq[3][in], vz_w);

            // 3. Dirichlet temperature: T_ghost = 2 * T_wall - T_in (only if compressible)
            if constexpr (NVars >= 5) {
                numerics::boundary::apply_dirichlet_value(lq[4][gh], lq[4][in], tmp_w);
                lq[4][gh] = std::max(lq[4][gh], 1.0);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Isothermal No-Slip Wall.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void no_slip_wall_grad_kernel(std::span<const double* const> q,
                                     std::span<double* const> gx,
                                     std::span<double* const> gy,
                                     std::span<double* const> gz,
                                     const mesh::MeshPart& m,
                                     const LocalIndex fbeg,
                                     const LocalIndex fend,
                                     const NoSlipWallParams& p) noexcept {
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

    const double q_wall[4] = {p.vx_wall, p.vy_wall, p.vz_wall, p.tmp_wall};

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // 1. Zero-order extrapolation for pressure gradient: dp/dn = 0
            numerics::boundary::apply_extrapolate_gradient(
                lgx[0][gh], lgy[0][gh], lgz[0][gh],
                lgx[0][in], lgy[0][in], lgz[0][in]);

            // 2. Fixed value gradients for velocities (1..3) and temperature (4 if NVars == 5)
            for (std::size_t d = 0; d < NVars - 1; ++d) {
                const std::size_t v = 1 + d;
                numerics::boundary::apply_dirichlet_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in],
                    lq[v][in], q_wall[d],
                    nx, ny, nz, rcfn_inv);
            }
        });
}

} // namespace kernels

/**
 * @class NoSlipWallBC
 * @brief Isothermal No-Slip Wall boundary condition implementation.
 * Gracefully acts as standard solid wall for incompressible solvers (skipping temperature).
 */
class NoSlipWallBC final : public BoundaryCondition {
public:
    NoSlipWallBC(std::string zone,
                 const LocalIndex fbeg,
                 const LocalIndex fend,
                 const NoSlipWallParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "NoSlipWallBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::no_slip_wall_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::no_slip_wall_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
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
            kernels::no_slip_wall_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::no_slip_wall_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void apply_momentum_bc(const mesh::MeshPart& mesh,
                           const mesh::MeshAuxGeometry& aux_geom,
                           double* CFD_RESTRICT diag,
                           double* CFD_RESTRICT rhs_u,
                           double* CFD_RESTRICT rhs_v,
                           double* CFD_RESTRICT rhs_w,
                           double* CFD_RESTRICT m_dot,
                           const double rho,
                           const double mu,
                           const double* CFD_RESTRICT mut = nullptr) const override {
        static_cast<void>(rho);

        const double uw = m_p.vx_wall;
        const double vw = m_p.vy_wall;
        const double ww = m_p.vz_wall;

        const double* CFD_RESTRICT dist_inv = aux_geom.face_cell_dist_inv.data();

        for (LocalIndex face_idx = m_begin; face_idx < m_end; ++face_idx) {
            const auto f = static_cast<std::size_t>(face_idx);
            const auto owner = static_cast<std::size_t>(mesh.face_owner[f]);

            if (m_dot != nullptr) {
                m_dot[f] = 0.0;
            }

            const double mu_eff = mu + (mut ? mut[owner] : 0.0);
            const double area   = mesh.face_area[f];
            const double D_b    = mu_eff * area * dist_inv[f];

            diag[owner] += D_b;

            if (uw != 0.0) rhs_u[owner] += D_b * uw;
            if (vw != 0.0) rhs_v[owner] += D_b * vw;
            if (ww != 0.0) rhs_w[owner] += D_b * ww;
        }
    }

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWall; }

private:
    NoSlipWallParams m_p;
};

} // namespace cfd::bc::physical
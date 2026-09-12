#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/bc/bc_fill_gradients.hpp"
#include "cfd/solver/bc/bc_fill_values.hpp"
#include "cfd/solver/eos/concepts.hpp"

namespace cfd::solver::bc {

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

    static NoSlipWallHeatFluxParams adiabatic() noexcept {
        return NoSlipWallHeatFluxParams{0.0, 0.0, 0.0, 0.0};
    }

    static NoSlipWallHeatFluxParams fixed_gradient(const double grad_T) noexcept {
        return NoSlipWallHeatFluxParams{0.0, 0.0, 0.0, grad_T};
    }

    static NoSlipWallHeatFluxParams moving_gradient(const double u,
                                                    const double v,
                                                    const double w,
                                                    const double grad_T) noexcept {
        return NoSlipWallHeatFluxParams{u, v, w, grad_T};
    }
};

namespace {

/** @brief Fills ghost cells with state values for No-Slip Wall with specified temperature gradient */
inline void no_slip_wall_heat_flux_kernel(std::span<double* const> q,
                                          const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend,
                                          const NoSlipWallHeatFluxParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology and metric arrays with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr         = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr         = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr         = m.face_normal_z.data();

    const double* CFD_RESTRICT fcx_ptr = m.face_centroid_x.data();
    const double* CFD_RESTRICT fcy_ptr = m.face_centroid_y.data();
    const double* CFD_RESTRICT fcz_ptr = m.face_centroid_z.data();

    const double* CFD_RESTRICT ccx_ptr = m.cell_centroid_x.data();
    const double* CFD_RESTRICT ccy_ptr = m.cell_centroid_y.data();
    const double* CFD_RESTRICT ccz_ptr = m.cell_centroid_z.data();

    // Cache primitive field pointers
    double* CFD_RESTRICT prs = q[0];
    double* CFD_RESTRICT vx  = q[1];
    double* CFD_RESTRICT vy  = q[2];
    double* CFD_RESTRICT vz  = q[3];
    double* CFD_RESTRICT tmp = q[4];

    // Cache wall parameters in registers
    const double vx_w          = p.vx_wall;
    const double vy_w          = p.vy_wall;
    const double vz_w          = p.vz_wall;
    const double tmp_grad_wall = p.tmp_grad_wall;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // 1. Zero normal pressure gradient: dp/dn = 0
        apply_extrapolation0_bc(prs[gh], prs[in]);

        // 2. Dirichlet velocity: v_ghost = 2 * v_wall - v_in
        apply_fixed_value_bc(vx[gh], vx[in], vx_w);
        apply_fixed_value_bc(vy[gh], vy[in], vy_w);
        apply_fixed_value_bc(vz[gh], vz[in], vz_w);

        // 3. Neumann temperature gradient: dT/dn = tmp_grad_wall
        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        const double fcx = fcx_ptr[face_idx];
        const double fcy = fcy_ptr[face_idx];
        const double fcz = fcz_ptr[face_idx];

        const double ccx = ccx_ptr[in];
        const double ccy = ccy_ptr[in];
        const double ccz = ccz_ptr[in];

        const double rcfx = fcx - ccx;
        const double rcfy = fcy - ccy;
        const double rcfz = fcz - ccz;

        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;

        apply_fixed_gradient_bc(tmp[gh], tmp[in], tmp_grad_wall, rcfn);

        // Physical lower bound guard against negative temperature in strong expansion
        tmp[gh] = std::max(tmp[gh], 1.0);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for No-Slip Wall with specified temperature gradient */
inline void no_slip_wall_heat_flux_grad_kernel(std::span<const double* const> q,
                                               std::span<double* const> gx,
                                               std::span<double* const> gy,
                                               std::span<double* const> gz,
                                               const mesh::MeshPart& m,
                                               const LocalIndex fbeg,
                                               const LocalIndex fend,
                                               const NoSlipWallHeatFluxParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology and metric arrays with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr         = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr         = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr         = m.face_normal_z.data();

    const double* CFD_RESTRICT fcx_ptr = m.face_centroid_x.data();
    const double* CFD_RESTRICT fcy_ptr = m.face_centroid_y.data();
    const double* CFD_RESTRICT fcz_ptr = m.face_centroid_z.data();

    const double* CFD_RESTRICT ccx_ptr = m.cell_centroid_x.data();
    const double* CFD_RESTRICT ccy_ptr = m.cell_centroid_y.data();
    const double* CFD_RESTRICT ccz_ptr = m.cell_centroid_z.data();

    const double v_wall[3]     = {p.vx_wall, p.vy_wall, p.vz_wall};
    const double tmp_grad_wall = p.tmp_grad_wall;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        const double fcx = fcx_ptr[face_idx];
        const double fcy = fcy_ptr[face_idx];
        const double fcz = fcz_ptr[face_idx];

        const double ccx = ccx_ptr[in];
        const double ccy = ccy_ptr[in];
        const double ccz = ccz_ptr[in];

        const double rcfx = fcx - ccx;
        const double rcfy = fcy - ccy;
        const double rcfz = fcz - ccz;

        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
        const double rcfn_inv = 1.0 / std::max(rcfn, 1.0e-14);

        // 1. Pressure gradient extrapolation
        apply_grad_extrapolation0_bc(gx[0][gh], gy[0][gh], gz[0][gh],
                                     gx[0][in], gy[0][in], gz[0][in]);

        // 2. Velocity gradient (fixed value at wall)
        for (std::size_t d = 0; d < 3; ++d) {
            const std::size_t v = 1 + d;
            apply_grad_fixed_value_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                      gx[v][in], gy[v][in], gz[v][in],
                                      q[v][in], v_wall[d], nx, ny, nz, rcfn_inv);
        }

        // 3. Temperature gradient (fixed normal gradient)
        apply_grad_fixed_gradient_bc(gx[4][gh], gy[4][gh], gz[4][gh],
                                     gx[4][in], gy[4][in], gz[4][in],
                                     tmp_grad_wall, nx, ny, nz);

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class NoSlipWallHeatFluxBC
 * @brief No-Slip Wall boundary condition with specified normal temperature gradient / heat flux.
 */
template <eos::EquationOfStatePolicy EOS>
class NoSlipWallHeatFluxBC final : public BoundaryCondition<EOS> {
public:
    NoSlipWallHeatFluxBC(std::string zone,
                         const LocalIndex fbeg,
                         const LocalIndex fend,
                         const NoSlipWallHeatFluxParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "NoSlipWallHeatFluxBC requires exactly 5 mean-flow variables");
        no_slip_wall_heat_flux_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        no_slip_wall_heat_flux_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWallHeatFlux; }

private:
    NoSlipWallHeatFluxParams m_p;
};

} // namespace cfd::solver::bc
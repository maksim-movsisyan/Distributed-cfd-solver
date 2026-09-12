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
 * @brief Parameters for Isothermal No-Slip Wall Boundary Condition.
 */
struct NoSlipWallParams {
    double vx_wall{0.0};     ///< Wall velocity X [m/s]
    double vy_wall{0.0};     ///< Wall velocity Y [m/s]
    double vz_wall{0.0};     ///< Wall velocity Z [m/s]
    double tmp_wall{288.15}; ///< Wall temperature [K]

    static NoSlipWallParams stationary_isothermal(const double T_wall) noexcept {
        return NoSlipWallParams{0.0, 0.0, 0.0, T_wall};
    }

    static NoSlipWallParams moving_isothermal(const double u,
                                             const double v,
                                             const double w,
                                             const double T_wall) noexcept {
        return NoSlipWallParams{u, v, w, T_wall};
    }
};

namespace {

/** @brief Fills ghost cells with state values for Isothermal No-Slip Wall */
inline void no_slip_wall_kernel(std::span<double* const> q,
                                const mesh::MeshPart& m,
                                const LocalIndex fbeg,
                                const LocalIndex fend,
                                const NoSlipWallParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology array with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();

    // Cache primitive field pointers
    double* CFD_RESTRICT prs = q[0];
    double* CFD_RESTRICT vx  = q[1];
    double* CFD_RESTRICT vy  = q[2];
    double* CFD_RESTRICT vz  = q[3];
    double* CFD_RESTRICT tmp = q[4];

    // Cache wall parameters in registers
    const double vx_w  = p.vx_wall;
    const double vy_w  = p.vy_wall;
    const double vz_w  = p.vz_wall;
    const double tmp_w = p.tmp_wall;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // 1. Zero normal pressure gradient: dp/dn = 0
        apply_extrapolation0_bc(prs[gh], prs[in]);

        // 2. Dirichlet velocity: v_ghost = 2 * v_wall - v_in
        apply_fixed_value_bc(vx[gh], vx[in], vx_w);
        apply_fixed_value_bc(vy[gh], vy[in], vy_w);
        apply_fixed_value_bc(vz[gh], vz[in], vz_w);

        // 3. Dirichlet temperature: T_ghost = 2 * T_wall - T_in
        apply_fixed_value_bc(tmp[gh], tmp[in], tmp_w);

        // Physical lower bound guard against negative temperature
        tmp[gh] = std::max(tmp[gh], 1.0);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Isothermal No-Slip Wall */
inline void no_slip_wall_grad_kernel(std::span<const double* const> q,
                                     std::span<double* const> gx,
                                     std::span<double* const> gy,
                                     std::span<double* const> gz,
                                     const mesh::MeshPart& m,
                                     const LocalIndex fbeg,
                                     const LocalIndex fend,
                                     const NoSlipWallParams& p) noexcept {
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

    // Wall target Dirichlet values: [vx, vy, vz, tmp]
    const double q_wall[4] = {p.vx_wall, p.vy_wall, p.vz_wall, p.tmp_wall};

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

        // 1. Zero-order extrapolation for pressure gradient: dp/dn = 0
        apply_grad_extrapolation0_bc(gx[0][gh], gy[0][gh], gz[0][gh],
                                     gx[0][in], gy[0][in], gz[0][in]);

        // 2. Fixed value gradient for velocities (1..3) and temperature (4)
        for (std::size_t d = 0; d < 4; ++d) {
            const std::size_t v = 1 + d;
            apply_grad_fixed_value_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                      gx[v][in], gy[v][in], gz[v][in],
                                      q[v][in], q_wall[d], nx, ny, nz, rcfn_inv);
        }

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class NoSlipWallBC
 * @brief Isothermal No-Slip Wall boundary condition implementation.
 */
template <eos::EquationOfStatePolicy EOS>
class NoSlipWallBC final : public BoundaryCondition<EOS> {
public:
    NoSlipWallBC(std::string zone,
                 const LocalIndex fbeg,
                 const LocalIndex fend,
                 const NoSlipWallParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "NoSlipWallBC requires exactly 5 mean-flow variables");
        no_slip_wall_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        no_slip_wall_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWall; }

private:
    NoSlipWallParams m_p;
};

} // namespace cfd::solver::bc
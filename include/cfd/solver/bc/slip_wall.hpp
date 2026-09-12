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

namespace {

/** @brief Fills ghost cells with state values for Slip / Symmetry boundary */
inline void slip_wall_kernel(std::span<double* const> q,
                             const mesh::MeshPart& m,
                             const LocalIndex fbeg,
                             const LocalIndex fend) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology and normal arrays with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr         = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr         = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr         = m.face_normal_z.data();

    // Cache primitive field pointers
    double* CFD_RESTRICT prs = q[0];
    double* CFD_RESTRICT vx  = q[1];
    double* CFD_RESTRICT vy  = q[2];
    double* CFD_RESTRICT vz  = q[3];
    double* CFD_RESTRICT tmp = q[4];

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        // 1. Reflect velocity vector (Symmetry / Slip: u_n = 0)
        apply_slip_component_bc(vx[gh], vy[gh], vz[gh],
                                vx[in], vy[in], vz[in],
                                nx, ny, nz);

        // 2. Extrapolate scalars from inner cell (dp/dn = 0, dT/dn = 0)
        apply_extrapolation0_bc(prs[gh], prs[in]);
        apply_extrapolation0_bc(tmp[gh], tmp[in]);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Slip / Symmetry boundary */
inline void slip_wall_grad_kernel(std::span<const double* const> q,
                                  std::span<double* const> gx,
                                  std::span<double* const> gy,
                                  std::span<double* const> gz,
                                  const mesh::MeshPart& m,
                                  const LocalIndex fbeg,
                                  const LocalIndex fend) noexcept {
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

        // 1. Pressure gradient (zero normal gradient: dp/dn = 0)
        apply_grad_fixed_gradient_bc(gx[0][gh], gy[0][gh], gz[0][gh],
                                     gx[0][in], gy[0][in], gz[0][in],
                                     0.0, nx, ny, nz);

        // 2. Velocity gradients (slip reflection)
        for (std::size_t d = 0; d < 3; ++d) {
            const std::size_t v = 1 + d;
            apply_grad_slip_component_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                         gx[v][in], gy[v][in], gz[v][in],
                                         q[v][in], q[v][gh], nx, ny, nz, rcfn_inv);
        }

        // 3. Temperature gradient (zero normal gradient / adiabatic: dT/dn = 0)
        apply_grad_fixed_gradient_bc(gx[4][gh], gy[4][gh], gz[4][gh],
                                     gx[4][in], gy[4][in], gz[4][in],
                                     0.0, nx, ny, nz);

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class SlipWallBC
 * @brief Slip wall boundary condition implementation (inviscid wall / symmetry plane).
 */
template <eos::EquationOfStatePolicy EOS>
class SlipWallBC final : public BoundaryCondition<EOS> {
public:
    SlipWallBC(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "SlipWallBC requires exactly 5 mean-flow variables");
        slip_wall_kernel(q, mesh, this->m_begin, this->m_end);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        slip_wall_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SlipWall; }
};

} // namespace cfd::solver::bc
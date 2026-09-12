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
 * @brief Parameters for Subsonic Outlet (imposed static backpressure).
 */
struct SubsonicOutletParams {
    double prs_outlet{101325.0}; ///< Target static backpressure p_back [Pa]

    static SubsonicOutletParams from_pressure(const double p_back) noexcept {
        return SubsonicOutletParams{p_back};
    }
};

namespace {

/** @brief Fills ghost cells with state values for Subsonic Outlet */
inline void subsonic_outlet_kernel(std::span<double* const> q,
                                   const mesh::MeshPart& m,
                                   const LocalIndex fbeg,
                                   const LocalIndex fend,
                                   const SubsonicOutletParams& p) noexcept {
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

    // Cache outlet backpressure in register
    const double prs_outlet = p.prs_outlet;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // 1. Fixed Dirichlet static backpressure: p_face = p_outlet
        apply_fixed_value_bc(prs[gh], prs[in], prs_outlet);

        // Physical lower bound guard against negative pressure
        prs[gh] = std::max(prs[gh], 1.0);

        // 2. Extrapolate velocity components from interior (Neumann: dv/dn = 0)
        apply_extrapolation0_bc(vx[gh], vx[in]);
        apply_extrapolation0_bc(vy[gh], vy[in]);
        apply_extrapolation0_bc(vz[gh], vz[in]);

        // 3. Extrapolate temperature from interior (Neumann: dT/dn = 0)
        apply_extrapolation0_bc(tmp[gh], tmp[in]);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Subsonic Outlet */
inline void subsonic_outlet_grad_kernel(std::span<const double* const> q,
                                        std::span<double* const> gx,
                                        std::span<double* const> gy,
                                        std::span<double* const> gz,
                                        const mesh::MeshPart& m,
                                        const LocalIndex fbeg,
                                        const LocalIndex fend,
                                        const SubsonicOutletParams& p) noexcept {
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

    const double prs_outlet = p.prs_outlet;

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

        // 1. Fixed value pressure gradient
        apply_grad_fixed_value_bc(gx[0][gh], gy[0][gh], gz[0][gh],
                                  gx[0][in], gy[0][in], gz[0][in],
                                  q[0][in], prs_outlet,
                                  nx, ny, nz, rcfn_inv);

        // 2. Extrapolate velocity (1..3) and temperature (4) gradients (Neumann: d(grad)/dn = 0)
        for (std::size_t v = 1; v < 5; ++v) {
            apply_grad_extrapolation0_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                         gx[v][in], gy[v][in], gz[v][in]);
        }

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class SubsonicOutletBC
 * @brief Subsonic Outlet boundary condition with imposed static backpressure.
 */
template <solver::eos::EquationOfStatePolicy EOS>
class SubsonicOutletBC final : public BoundaryCondition<EOS> {
public:
    SubsonicOutletBC(std::string zone,
                     const LocalIndex fbeg,
                     const LocalIndex fend,
                     const SubsonicOutletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "SubsonicOutletBC requires exactly 5 mean-flow variables");
        subsonic_outlet_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        subsonic_outlet_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicOutlet; }

private:
    SubsonicOutletParams m_p;
};

} // namespace cfd::solver::bc
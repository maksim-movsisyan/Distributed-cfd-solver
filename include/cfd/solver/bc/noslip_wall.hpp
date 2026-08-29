#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/bc/bc_fill_gradients.hpp"
#include "cfd/solver/bc/bc_fill_values.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/fields/fields_view.hpp"

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

/** @brief Set value in ghost cell for No-Slip Isothermal wall */
inline void no_slip_wall_kernel(fields::PrimitiveView<double> s,
                                const mesh::MeshPart& m,
                                const LocalIndex fbeg,
                                const LocalIndex fend,
                                const NoSlipWallParams& p) noexcept {
    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const auto in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const auto gh = n_cells + f_loc;                                  // ghost cell

        // ==== 1. Zero normal pressure gradient: dp/dn = 0 ====
        apply_extrapolation0_bc(s.prs[gh], s.prs[in]);

        // ==== 2. Dirichlet velocity: v_ghost = 2 * v_wall - v_in ====
        apply_fixed_value_bc(s.vx[gh], s.vx[in], p.vx_wall);
        apply_fixed_value_bc(s.vy[gh], s.vy[in], p.vy_wall);
        apply_fixed_value_bc(s.vz[gh], s.vz[in], p.vz_wall);

        // ==== 3. Dirichlet temperature (Isothermal wall): T_ghost = 2 * T_wall - T_in ====
        apply_fixed_value_bc(s.tmp[gh], s.tmp[in], p.tmp_wall);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for No-Slip Isothermal wall */
inline void no_slip_wall_grad_kernel(fields::ConstPrimitiveView s,
                                     fields::PrimitiveGradView<double> s_grad,
                                     const mesh::MeshPart& m,
                                     const LocalIndex fbeg,
                                     const LocalIndex fend,
                                     const NoSlipWallParams& p) noexcept {
    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const auto in = static_cast<std::size_t>(m.face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        const double nx = m.face_normal_x[face_idx];
        const double ny = m.face_normal_y[face_idx];
        const double nz = m.face_normal_z[face_idx];

        const double fcx = m.face_centroid_x[face_idx];
        const double fcy = m.face_centroid_y[face_idx];
        const double fcz = m.face_centroid_z[face_idx];

        const double ccx = m.cell_centroid_x[in];
        const double ccy = m.cell_centroid_y[in];
        const double ccz = m.cell_centroid_z[in];

        const double rcfx = fcx - ccx;
        const double rcfy = fcy - ccy;
        const double rcfz = fcz - ccz;

        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
        const double rcfn_inv = 1.0 / rcfn;

        // ==== 1. Zero-order extrapolation for pressure gradient ====
        apply_grad_extrapolation0_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                     s_grad.dprs_dx(in), s_grad.dprs_dy(in), s_grad.dprs_dz(in));

        // ==== 2. Fixed value gradient for velocities ====
        double gx = s_grad.dvx_dx(in), gy = s_grad.dvx_dy(in), gz = s_grad.dvx_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                  gx, gy, gz, s.vx[in], p.vx_wall, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvy_dx(in); gy = s_grad.dvy_dy(in); gz = s_grad.dvy_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                  gx, gy, gz, s.vy[in], p.vy_wall, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvz_dx(in); gy = s_grad.dvz_dy(in); gz = s_grad.dvz_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                  gx, gy, gz, s.vz[in], p.vz_wall, nx, ny, nz, rcfn_inv);

        // ==== 3. Fixed value gradient for temperature ====
        gx = s_grad.dtmp_dx(in); gy = s_grad.dtmp_dy(in); gz = s_grad.dtmp_dz(in);
        apply_grad_fixed_value_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                  gx, gy, gz, s.tmp[in], p.tmp_wall, nx, ny, nz, rcfn_inv);

        ++f_loc;
    }
}

/**
 * @class NoSlipWallBC
 * @brief Isothermal No-Slip Wall boundary condition implementation.
 */
template <eos::EquationOfState EOS>
class NoSlipWallBC final : public BoundaryCondition<EOS> {
public:
    NoSlipWallBC(std::string zone,
                 const LocalIndex fbeg,
                 const LocalIndex fend,
                 const NoSlipWallParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        no_slip_wall_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        no_slip_wall_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWall; }

private:
    NoSlipWallParams m_p;
};

} // namespace cfd::solver::bc
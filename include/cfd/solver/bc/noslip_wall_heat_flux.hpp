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

/** @brief Set value in ghost cell for No-Slip Wall with temperature gradient */
inline void no_slip_wall_heat_flux_kernel(fields::PrimitiveView<double> s,
                                          const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend,
                                          const NoSlipWallHeatFluxParams& p) noexcept {
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

        // ==== 3. Neumann temperature gradient: dT/dn = tmp_grad_wall ====
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

        apply_fixed_gradient_bc(s.tmp[gh], s.tmp[in], p.tmp_grad_wall, rcfn);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for No-Slip Wall with temperature gradient */
inline void no_slip_wall_heat_flux_grad_kernel(fields::ConstPrimitiveView s,
                                               fields::PrimitiveGradView<double> s_grad,
                                               const mesh::MeshPart& m,
                                               const LocalIndex fbeg,
                                               const LocalIndex fend,
                                               const NoSlipWallHeatFluxParams& p) noexcept {
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

        // ==== 1. Pressure gradient extrapolation ====
        apply_grad_extrapolation0_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                     s_grad.dprs_dx(in), s_grad.dprs_dy(in), s_grad.dprs_dz(in));

        // ==== 2. Velocity gradient (fixed value at face) ====
        double gx = s_grad.dvx_dx(in), gy = s_grad.dvx_dy(in), gz = s_grad.dvx_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                  gx, gy, gz, s.vx[in], p.vx_wall, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvy_dx(in); gy = s_grad.dvy_dy(in); gz = s_grad.dvy_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                  gx, gy, gz, s.vy[in], p.vy_wall, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvz_dx(in); gy = s_grad.dvz_dy(in); gz = s_grad.dvz_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                  gx, gy, gz, s.vz[in], p.vz_wall, nx, ny, nz, rcfn_inv);

        // ==== 3. Temperature gradient (fixed normal gradient) ====
        gx = s_grad.dtmp_dx(in); gy = s_grad.dtmp_dy(in); gz = s_grad.dtmp_dz(in);
        apply_grad_fixed_gradient_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                     gx, gy, gz, p.tmp_grad_wall, nx, ny, nz);

        ++f_loc;
    }
}

/**
 * @class NoSlipWallHeatFluxBC
 * @brief No-Slip Wall boundary condition with specified normal temperature gradient / heat flux.
 */
template <eos::EquationOfState EOS>
class NoSlipWallHeatFluxBC final : public BoundaryCondition<EOS> {
public:
    NoSlipWallHeatFluxBC(std::string zone,
                         const LocalIndex fbeg,
                         const LocalIndex fend,
                         const NoSlipWallHeatFluxParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        no_slip_wall_heat_flux_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        no_slip_wall_heat_flux_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::NoSlipWallHeatFlux; }

private:
    NoSlipWallHeatFluxParams m_p;
};

} // namespace cfd::solver::bc
#pragma once

#include <cmath>
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
 * @brief Canonical resolved primitive state for Subsonic Inlet (fixed velocity & temperature, extrapolated pressure).
 * @note In from_mach_angles and from_mach_direction IDEAL GAS EOS SUGESTED!!!!
 */
struct SubsonicInletParams {    
    double vx_inlet{0.0};
    double vy_inlet{0.0};
    double vz_inlet{0.0};
    double tmp_inlet{288.15};

    // 1. Direct velocity components (u, v, w) + temperature T
    static SubsonicInletParams from_velocities(const double u,
                                               const double v,
                                               const double w,
                                               const double T) noexcept {
        return SubsonicInletParams{u, v, w, T};
    }

    // 2. Mach number + aerodynamic angles (alpha, beta in degrees) + temperature T
    template <eos::EquationOfState EOS>
    static SubsonicInletParams from_mach_angles(const EOS& eos,
                                                const double T,
                                                const double mach,
                                                const double alpha_deg,
                                                const double beta_deg) noexcept {
        constexpr double kDegToRad = M_PI / 180.0;
        const double alpha_rad = alpha_deg * kDegToRad;
        const double beta_rad  = beta_deg  * kDegToRad;

        const double rho = eos.density_Tp(T, 101325.0);
        const double a   = eos.sound_speed_rhop(rho, 101325.0);
        const double v_mag = mach * a;

        const double u = v_mag * std::cos(alpha_rad) * std::cos(beta_rad);
        const double v = -v_mag * std::sin(beta_rad);
        const double w = v_mag * std::sin(alpha_rad) * std::cos(beta_rad);

        return SubsonicInletParams{u, v, w, T};
    }

    // 3. Mach number + unit direction vector (dx, dy, dz) + temperature T
    template <eos::EquationOfState EOS>
    static SubsonicInletParams from_mach_direction(const EOS& eos,
                                                   const double T,
                                                   const double mach,
                                                   const double dx,
                                                   const double dy,
                                                   const double dz) noexcept {
        const double rho = eos.density_Tp(T, 101325.0);
        const double a   = eos.sound_speed_rhop(rho, 101325.0);
        const double v_mag = mach * a;

        const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_norm = (norm > 1.0e-14) ? (1.0 / norm) : 0.0;

        return SubsonicInletParams{
            v_mag * dx * inv_norm,
            v_mag * dy * inv_norm,
            v_mag * dz * inv_norm,
            T
        };
    }
};

/** @brief Set value in ghost cell for Subsonic Inlet */
inline void subsonic_inlet_kernel(fields::PrimitiveView<double> s,
                                  const mesh::MeshPart& m,
                                  const LocalIndex fbeg,
                                  const LocalIndex fend,
                                  const SubsonicInletParams& p) noexcept {
    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const auto in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const auto gh = n_cells + f_loc;                                  // ghost cell

        // ==== 1. Extrapolate pressure from interior: dp/dn = 0 ====
        apply_extrapolation0_bc(s.prs[gh], s.prs[in]);

        // ==== 2. Fixed Dirichlet velocities ====
        apply_fixed_value_bc(s.vx[gh], s.vx[in], p.vx_inlet);
        apply_fixed_value_bc(s.vy[gh], s.vy[in], p.vy_inlet);
        apply_fixed_value_bc(s.vz[gh], s.vz[in], p.vz_inlet);

        // ==== 3. Fixed Dirichlet temperature ====
        apply_fixed_value_bc(s.tmp[gh], s.tmp[in], p.tmp_inlet);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for Subsonic Inlet */
inline void subsonic_inlet_grad_kernel(fields::ConstPrimitiveView s,
                                       fields::PrimitiveGradView<double> s_grad,
                                       const mesh::MeshPart& m,
                                       const LocalIndex fbeg,
                                       const LocalIndex fend,
                                       const SubsonicInletParams& p) noexcept {
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

        // ==== 1. Extrapolate pressure gradient ====
        apply_grad_extrapolation0_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                     s_grad.dprs_dx(in), s_grad.dprs_dy(in), s_grad.dprs_dz(in));

        // ==== 2. Fixed value gradients for velocities ====
        double gx = s_grad.dvx_dx(in), gy = s_grad.dvx_dy(in), gz = s_grad.dvx_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                  gx, gy, gz, s.vx[in], p.vx_inlet, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvy_dx(in); gy = s_grad.dvy_dy(in); gz = s_grad.dvy_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                  gx, gy, gz, s.vy[in], p.vy_inlet, nx, ny, nz, rcfn_inv);

        gx = s_grad.dvz_dx(in); gy = s_grad.dvz_dy(in); gz = s_grad.dvz_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                  gx, gy, gz, s.vz[in], p.vz_inlet, nx, ny, nz, rcfn_inv);

        // ==== 3. Fixed value gradient for temperature ====
        gx = s_grad.dtmp_dx(in); gy = s_grad.dtmp_dy(in); gz = s_grad.dtmp_dz(in);
        apply_grad_fixed_value_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                  gx, gy, gz, s.tmp[in], p.tmp_inlet, nx, ny, nz, rcfn_inv);

        ++f_loc;
    }
}

/**
 * @class SubsonicInletBC
 * @brief Subsonic Inlet boundary condition implementation.
 */
template <eos::EquationOfState EOS>
class SubsonicInletBC final : public BoundaryCondition<EOS> {
public:
    SubsonicInletBC(std::string zone,
                    const LocalIndex fbeg,
                    const LocalIndex fend,
                    const SubsonicInletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        subsonic_inlet_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        subsonic_inlet_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicInlet; }

private:
    SubsonicInletParams m_p;
};

} // namespace cfd::solver::bc
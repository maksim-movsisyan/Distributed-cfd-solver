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
 * @brief Precomputed freestream parameters for Characteristic Farfield BC.
 */
struct FarfieldParams {
    double prs_inf{101325.0};   ///< Static pressure p_inf [Pa]
    double tmp_inf{288.15};     ///< Static temperature T_inf [K]
    double vx_inf{0.0};         ///< Freestream velocity-x [m/s]
    double vy_inf{0.0};         ///< Freestream velocity-y [m/s]
    double vz_inf{0.0};         ///< Freestream velocity-z [m/s]
    double gamma{1.4};          ///< Specific heat ratio [-]
    double R{287.052874};       ///< Specific gas constant [J / (kg K)]

    // if pressure, vlocity and tempareature are given
    template <eos::EquationOfState EOS>
    static FarfieldParams from_velocities(const EOS& eos,
                                          const double p,
                                          const double u,
                                          const double v,
                                          const double w,
                                          const double T) noexcept {
        return FarfieldParams{p, T, u, v, w, eos.gamma(), eos.gas_constant()};
    }

    // if pressure, mach, angel of atack, slip angel and temperature are given
    template <eos::EquationOfState EOS>
    static FarfieldParams from_mach_angles(const EOS& eos,
                                           const double p,
                                           const double T,
                                           const double mach,
                                           const double alpha_deg,
                                           const double beta_deg) noexcept {
        constexpr double kDegToRad = M_PI / 180.0;
        const double alpha_rad = alpha_deg * kDegToRad;
        const double beta_rad  = beta_deg  * kDegToRad;

        const double rho = eos.density_Tp(T, p);
        const double a   = eos.sound_speed_rhop(rho, p);
        const double v_mag = mach * a;

        const double u = v_mag * std::cos(alpha_rad) * std::cos(beta_rad);
        const double v = -v_mag * std::sin(beta_rad);
        const double w = v_mag * std::sin(alpha_rad) * std::cos(beta_rad);

        return FarfieldParams{p, T, u, v, w, eos.gamma(), eos.gas_constant()};
    }

    // if pressure, mach, direction vector and temperature are given
    template <eos::EquationOfState EOS>
    static FarfieldParams from_mach_direction(const EOS& eos,
                                              const double p,
                                              const double T,
                                              const double mach,
                                              const double dx,
                                              const double dy,
                                              const double dz) noexcept {
        const double rho = eos.density_Tp(T, p);
        const double a   = eos.sound_speed_rhop(rho, p);
        const double v_mag = mach * a;

        const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_norm = (norm > 1.0e-14) ? (1.0 / norm) : 0.0;

        return FarfieldParams{
            p, T,
            v_mag * dx * inv_norm,
            v_mag * dy * inv_norm,
            v_mag * dz * inv_norm,
            eos.gamma(),
            eos.gas_constant()
        };
    }
};

/** @brief Helper evaluating boundary face state from 1D Riemann Invariants */
inline void compute_riemann_farfield_state(const FarfieldParams& p,
                                          const double p_in, const double T_in,
                                          const double vx_in, const double vy_in, const double vz_in,
                                          const double nx, const double ny, const double nz,
                                          double& p_b, double& T_b,
                                          double& vx_b, double& vy_b, double& vz_b,
                                          bool& is_supersonic_outflow) noexcept {
    const double gm1 = p.gamma - 1.0;
    const double inv_gm1 = 1.0 / gm1;

    // Normal velocities
    const double un_in  = vx_in * nx + vy_in * ny + vz_in * nz;
    const double un_inf = p.vx_inf * nx + p.vy_inf * ny + p.vz_inf * nz;

    // Speed of sound
    const double a_in  = std::sqrt(p.gamma * p.R * std::max(T_in, 1.0e-6));
    const double a_inf = std::sqrt(p.gamma * p.R * std::max(p.tmp_inf, 1.0e-6));

    const double mn_in = un_in / a_in;

    // 1. Supersonic Outflow: full extrapolation
    if (mn_in >= 1.0) {
        is_supersonic_outflow = true;
        p_b = p_in;
        T_b = T_in;
        vx_b = vx_in;
        vy_b = vy_in;
        vz_b = vz_in;
        return;
    }

    is_supersonic_outflow = false;

    // 2. Supersonic Inflow: full freestream Dirichlet
    if (mn_in <= -1.0) {
        p_b = p.prs_inf;
        T_b = p.tmp_inf;
        vx_b = p.vx_inf;
        vy_b = p.vy_inf;
        vz_b = p.vz_inf;
        return;
    }

    // 3 & 4. Subsonic Inflow / Outflow: Riemann Invariants
    const double R_out = un_in + 2.0 * a_in * inv_gm1;
    const double R_in  = un_inf - 2.0 * a_inf * inv_gm1;

    const double un_b = 0.5 * (R_out + R_in);
    double a_b = 0.25 * gm1 * (R_out - R_in);
    if (a_b <= 0.0) a_b = a_inf; // Numerical safeguard

    double entropy_s = 0.0;

    if (un_b >= 0.0) {
        // Subsonic Outflow: tangential velocity and entropy from interior
        vx_b = vx_in + (un_b - un_in) * nx;
        vy_b = vy_in + (un_b - un_in) * ny;
        vz_b = vz_in + (un_b - un_in) * nz;

        const double rho_in = p_in / (p.R * std::max(T_in, 1.0e-6));
        entropy_s = p_in / std::pow(std::max(rho_in, 1.0e-12), p.gamma);
    } else {
        // Subsonic Inflow: tangential velocity and entropy from freestream
        vx_b = p.vx_inf + (un_b - un_inf) * nx;
        vy_b = p.vy_inf + (un_b - un_inf) * ny;
        vz_b = p.vz_inf + (un_b - un_inf) * nz;

        const double rho_inf = p.prs_inf / (p.R * std::max(p.tmp_inf, 1.0e-6));
        entropy_s = p.prs_inf / std::pow(std::max(rho_inf, 1.0e-12), p.gamma);
    }

    const double rho_b = std::pow((a_b * a_b) / (p.gamma * std::max(entropy_s, 1.0e-12)), inv_gm1);
    p_b = (rho_b * a_b * a_b) / p.gamma;
    T_b = (a_b * a_b) / (p.gamma * p.R);
}

/** @brief Set value in ghost cell for farfield */
inline void farfield_kernel(fields::PrimitiveView<double> s,
                            const mesh::MeshPart& m,
                            const LocalIndex fbeg,
                            const LocalIndex fend,
                            const FarfieldParams& p) noexcept {
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

        double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
        bool is_supersonic_outflow = false;

        compute_riemann_farfield_state(p, s.prs[in], s.tmp[in],
                                       s.vx[in], s.vy[in], s.vz[in],
                                       nx, ny, nz,
                                       pb, Tb, vxb, vyb, vzb,
                                       is_supersonic_outflow);

        if (is_supersonic_outflow) {
            apply_extrapolation0_bc(s.prs[gh], s.prs[in]);
            apply_extrapolation0_bc(s.vx[gh],  s.vx[in]);
            apply_extrapolation0_bc(s.vy[gh],  s.vy[in]);
            apply_extrapolation0_bc(s.vz[gh],  s.vz[in]);
            apply_extrapolation0_bc(s.tmp[gh], s.tmp[in]);
        } else {
            apply_fixed_value_bc(s.prs[gh], s.prs[in], pb);
            apply_fixed_value_bc(s.vx[gh],  s.vx[in],  vxb);
            apply_fixed_value_bc(s.vy[gh],  s.vy[in],  vyb);
            apply_fixed_value_bc(s.vz[gh],  s.vz[in],  vzb);
            apply_fixed_value_bc(s.tmp[gh], s.tmp[in], Tb);
        }

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for farfield */
inline void farfield_grad_kernel(fields::ConstPrimitiveView s,
                                 fields::PrimitiveGradView<double> s_grad,
                                 const mesh::MeshPart& m,
                                 const LocalIndex fbeg,
                                 const LocalIndex fend,
                                 const FarfieldParams& p) noexcept {
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

        double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
        bool is_supersonic_outflow = false;

        compute_riemann_farfield_state(p, s.prs[in], s.tmp[in],
                                       s.vx[in], s.vy[in], s.vz[in],
                                       nx, ny, nz,
                                       pb, Tb, vxb, vyb, vzb,
                                       is_supersonic_outflow);

        if (is_supersonic_outflow) {
            apply_grad_extrapolation0_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                         s_grad.dprs_dx(in), s_grad.dprs_dy(in), s_grad.dprs_dz(in));
            apply_grad_extrapolation0_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                         s_grad.dvx_dx(in), s_grad.dvx_dy(in), s_grad.dvx_dz(in));
            apply_grad_extrapolation0_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                         s_grad.dvy_dx(in), s_grad.dvy_dy(in), s_grad.dvy_dz(in));
            apply_grad_extrapolation0_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                         s_grad.dvz_dx(in), s_grad.dvz_dy(in), s_grad.dvz_dz(in));
            apply_grad_extrapolation0_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                         s_grad.dtmp_dx(in), s_grad.dtmp_dy(in), s_grad.dtmp_dz(in));
        } else {
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

            // Pressure
            double gx = s_grad.dprs_dx(in), gy = s_grad.dprs_dy(in), gz = s_grad.dprs_dz(in);
            apply_grad_fixed_value_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                      gx, gy, gz, s.prs[in], pb, nx, ny, nz, rcfn_inv);

            // Velocities
            gx = s_grad.dvx_dx(in); gy = s_grad.dvx_dy(in); gz = s_grad.dvx_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                      gx, gy, gz, s.vx[in], vxb, nx, ny, nz, rcfn_inv);

            gx = s_grad.dvy_dx(in); gy = s_grad.dvy_dy(in); gz = s_grad.dvy_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                      gx, gy, gz, s.vy[in], vyb, nx, ny, nz, rcfn_inv);

            gx = s_grad.dvz_dx(in); gy = s_grad.dvz_dy(in); gz = s_grad.dvz_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                      gx, gy, gz, s.vz[in], vzb, nx, ny, nz, rcfn_inv);

            // Temperature
            gx = s_grad.dtmp_dx(in); gy = s_grad.dtmp_dy(in); gz = s_grad.dtmp_dz(in);
            apply_grad_fixed_value_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                      gx, gy, gz, s.tmp[in], Tb, nx, ny, nz, rcfn_inv);
        }

        ++f_loc;
    }
}

/**
 * @class FarfieldBC
 * @brief Non-reflecting characteristic boundary condition based on 1D Riemann Invariants.
 */
template <eos::EquationOfState EOS>
class FarfieldBC final : public BoundaryCondition<EOS> {
public:
    FarfieldBC(std::string zone,
               const LocalIndex fbeg,
               const LocalIndex fend,
               const FarfieldParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend), m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        farfield_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        farfield_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::Farfield; }

private:
    FarfieldParams m_p;
};

} // namespace cfd::solver::bc
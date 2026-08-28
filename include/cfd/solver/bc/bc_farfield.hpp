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
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::bc {

/** 
 * @brief Precomputed freestream parameters for Characteristic Farfield BC.
 */
struct FarfieldParams {
    double prs_inf{101325.0};   ///< Static pressure p_inf [Pa]
    double tmp_inf{300.0};       ///< Static temperature T_inf [K]
    double vx_inf{0.0};         ///< Freestream velocity-x [m/s]
    double vy_inf{0.0};         ///< Freestream velocity-y [m/s]
    double vz_inf{0.0};         ///< Freestream velocity-z [m/s]
    double gamma{1.4};          ///< Specific heat ratio [-]
    double R{287.052874};       ///< Specific gas constant [J / (kg K)]
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
    const double a_in  = std::sqrt(p.gamma * p.R * T_in);
    const double a_inf = std::sqrt(p.gamma * p.R * p.tmp_inf);

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

        const double rho_in = p_in / (p.R * T_in);
        entropy_s = p_in / std::pow(rho_in, p.gamma);
    } else {
        // Subsonic Inflow: tangential velocity and entropy from freestream
        vx_b = p.vx_inf + (un_b - un_inf) * nx;
        vy_b = p.vy_inf + (un_b - un_inf) * ny;
        vz_b = p.vz_inf + (un_b - un_inf) * nz;

        const double rho_inf = p.prs_inf / (p.R * p.tmp_inf);
        entropy_s = p.prs_inf / std::pow(rho_inf, p.gamma);
    }

    const double rho_b = std::pow((a_b * a_b) / (p.gamma * entropy_s), inv_gm1);
    p_b = (rho_b * a_b * a_b) / p.gamma;
    T_b = (a_b * a_b) / (p.gamma * p.R);
}

/** @brief Set value in ghost cell */
inline void farfield_kernel(fields::PrimitiveView<double> s, const mesh::MeshPart& m,
                            LocalIndex fbeg, LocalIndex fend, const FarfieldParams p) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]);
        const std::size_t gh = n_cells + f_loc;

        const double nx = m.face_normal_x[face_idx];
        const double ny = m.face_normal_y[face_idx];
        const double nz = m.face_normal_z[face_idx];

        double pb, Tb, vxb, vyb, vzb;
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

/** @brief Set gradient in ghost cell */
inline void farfield_grad_kernel(fields::PrimitiveView<const double> s,
                                 fields::PrimitiveGradView<double> s_grad,
                                 const mesh::MeshPart& m,
                                 LocalIndex fbeg, LocalIndex fend,
                                 const FarfieldParams p) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]);
        const std::size_t gh = n_cells + f_loc;

        const double nx = m.face_normal_x[face_idx];
        const double ny = m.face_normal_y[face_idx];
        const double nz = m.face_normal_z[face_idx];

        double pb, Tb, vxb, vyb, vzb;
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
            double gx_in, gy_in, gz_in;

            // Pressure
            gx_in = s_grad.dprs_dx(in); gy_in = s_grad.dprs_dy(in); gz_in = s_grad.dprs_dz(in);
            apply_grad_fixed_value_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                      gx_in, gy_in, gz_in, s.prs[in], pb, nx, ny, nz, rcfn_inv);

            // Velocities
            gx_in = s_grad.dvx_dx(in); gy_in = s_grad.dvx_dy(in); gz_in = s_grad.dvx_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                      gx_in, gy_in, gz_in, s.vx[in], vxb, nx, ny, nz, rcfn_inv);

            gx_in = s_grad.dvy_dx(in); gy_in = s_grad.dvy_dy(in); gz_in = s_grad.dvy_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                      gx_in, gy_in, gz_in, s.vy[in], vyb, nx, ny, nz, rcfn_inv);

            gx_in = s_grad.dvz_dx(in); gy_in = s_grad.dvz_dy(in); gz_in = s_grad.dvz_dz(in);
            apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                      gx_in, gy_in, gz_in, s.vz[in], vzb, nx, ny, nz, rcfn_inv);

            // Temperature
            gx_in = s_grad.dtmp_dx(in); gy_in = s_grad.dtmp_dy(in); gz_in = s_grad.dtmp_dz(in);
            apply_grad_fixed_value_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                      gx_in, gy_in, gz_in, s.tmp[in], Tb, nx, ny, nz, rcfn_inv);
        }

        ++f_loc;
    }
}

/**
 * @class FarfieldBC
 * @brief Non-reflecting characteristic boundary condition based on 1D Riemann Invariants.
 */
class FarfieldBC : public BoundaryCondition {
public:
    /** @brief FarfieldBC constructor */
    FarfieldBC(std::string zone, LocalIndex fbeg, LocalIndex fend, const FarfieldParams p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    /** @brief FarfieldBC apply implementation */
    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh) const override {
        farfield_kernel(state, mesh, m_begin, m_end, m_p);
    }

    /** @brief FarfieldBC apply gradient implementation */
    void apply_grad(fields::PrimitiveView<const double> state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        farfield_grad_kernel(state, state_grad, mesh, m_begin, m_end, m_p);
    }

    BCType kind() const override { return BCType::Farfield; }

private:
    FarfieldParams m_p;
};

} // namespace cfd::solver::bc
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
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
    template <eos::EquationOfStatePolicy EOS>
    static FarfieldParams from_velocities(const EOS& eos,
                                          const double p,
                                          const double u,
                                          const double v,
                                          const double w,
                                          const double T) noexcept {
        return FarfieldParams{p, T, u, v, w, eos.gamma(), eos.gas_constant()};
    }

    // if pressure, mach, angel of atack, slip angel and temperature are given
    template <eos::EquationOfStatePolicy EOS>
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
    template <eos::EquationOfStatePolicy EOS>
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

namespace {

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
inline void farfield_kernel(std::span<double* const> q,
                            const mesh::MeshPart& m,
                            const LocalIndex fbeg,
                            const LocalIndex fend,
                            const FarfieldParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr         = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr         = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr         = m.face_normal_z.data();

    // Cache variable pointers
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

        double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
        bool is_supersonic_outflow = false;

        compute_riemann_farfield_state(p, prs[in], tmp[in],
                                       vx[in], vy[in], vz[in],
                                       nx, ny, nz,
                                       pb, Tb, vxb, vyb, vzb,
                                       is_supersonic_outflow);

        if (is_supersonic_outflow) {
            apply_extrapolation0_bc(prs[gh], prs[in]);
            apply_extrapolation0_bc(vx[gh],  vx[in]);
            apply_extrapolation0_bc(vy[gh],  vy[in]);
            apply_extrapolation0_bc(vz[gh],  vz[in]);
            apply_extrapolation0_bc(tmp[gh], tmp[in]);
        } else {
            apply_fixed_value_bc(prs[gh], prs[in], pb);
            apply_fixed_value_bc(vx[gh],  vx[in],  vxb);
            apply_fixed_value_bc(vy[gh],  vy[in],  vyb);
            apply_fixed_value_bc(vz[gh],  vz[in],  vzb);
            apply_fixed_value_bc(tmp[gh], tmp[in], Tb);

            // Numerical floor guards against strong expansion waves
            prs[gh] = std::max(prs[gh], 1.0);
            tmp[gh] = std::max(tmp[gh], 1.0);
        }
        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for farfield */
inline void farfield_grad_kernel(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& m,
                                 const LocalIndex fbeg,
                                 const LocalIndex fend,
                                 const FarfieldParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

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

    const double* CFD_RESTRICT prs = q[0];
    const double* CFD_RESTRICT vx  = q[1];
    const double* CFD_RESTRICT vy  = q[2];
    const double* CFD_RESTRICT vz  = q[3];
    const double* CFD_RESTRICT tmp = q[4];

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
        bool is_supersonic_outflow = false;

        compute_riemann_farfield_state(p, prs[in], tmp[in],
                                       vx[in], vy[in], vz[in],
                                       nx, ny, nz,
                                       pb, Tb, vxb, vyb, vzb,
                                       is_supersonic_outflow);

        if (is_supersonic_outflow) {

            for (std::size_t v = 0; v < 5; ++v) {
                apply_grad_extrapolation0_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                             gx[v][in], gy[v][in], gz[v][in]);
            }
        } else {
            const double rcfx = fcx_ptr[face_idx] - ccx_ptr[in];
            const double rcfy = fcy_ptr[face_idx] - ccy_ptr[in];
            const double rcfz = fcz_ptr[face_idx] - ccz_ptr[in];

            const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
            const double rcfn_inv = 1.0 / std::max(rcfn, 1.0e-14);

            const double q_b[5] = {pb, vxb, vyb, vzb, Tb};

            for (std::size_t v = 0; v < 5; ++v) {
                apply_grad_fixed_value_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                          gx[v][in], gy[v][in], gz[v][in], 
                                          q[v][in], q_b[v], nx, ny, nz, rcfn_inv);
            }
        }

        ++f_loc;
    }
}

} // namespace 

/**
 * @class FarfieldBC
 * @brief Non-reflecting characteristic boundary condition based on 1D Riemann Invariants.
 */
template <eos::EquationOfStatePolicy EOS>
class FarfieldBC final : public BoundaryCondition<EOS> {
public:
    FarfieldBC(std::string zone,
               const LocalIndex fbeg,
               const LocalIndex fend,
               const FarfieldParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q, 
                            const mesh::MeshPart& mesh,
                            const EOS&) const override {
        assert(q.size() == 5 && "Compressible FarfieldBC requires exactly 5 mean-flow variables");
        farfield_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q, 
                                 std::span<double* const> gx, 
                                 std::span<double* const> gy, 
                                 std::span<double* const> gz, 
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        farfield_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::Farfield; }

private:
    FarfieldParams m_p;
};

} // namespace cfd::solver::bc
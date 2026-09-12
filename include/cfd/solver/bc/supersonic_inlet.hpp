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
 * @brief Canonical resolved primitive state for Supersonic Inlet.
 */
struct SupersonicInletParams {
    double prs_inlet{101325.0};
    double vx_inlet{0.0};
    double vy_inlet{0.0};
    double vz_inlet{0.0};
    double tmp_inlet{288.15};

    // Direct primitive values: static pressure, velocity components (u, v, w), static temperature
    static SupersonicInletParams from_velocities(const double p,
                                                 const double u,
                                                 const double v,
                                                 const double w,
                                                 const double T) noexcept {
        return SupersonicInletParams{p, u, v, w, T};
    }

    // Mach number + unit direction vector (dx, dy, dz) + static pressure & temperature
    template <eos::EquationOfStatePolicy EOS>
    static SupersonicInletParams from_mach_direction(const EOS& eos,
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

        return SupersonicInletParams{
            p,
            v_mag * dx * inv_norm,
            v_mag * dy * inv_norm,
            v_mag * dz * inv_norm,
            T
        };
    }

    // Mach number + aerodynamic angles (alpha, beta in degrees) + static pressure & temperature
    template <eos::EquationOfStatePolicy EOS>
    static SupersonicInletParams from_mach_angles(const EOS& eos,
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

        return SupersonicInletParams{p, u, v, w, T};
    }
};

namespace {

/** @brief Fills ghost cells with state values for Supersonic Inlet */
inline void supersonic_inlet_kernel(std::span<double* const> q, 
                                    const mesh::MeshPart& m,
                                    const LocalIndex fbeg, 
                                    const LocalIndex fend, 
                                    const SupersonicInletParams& p) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology array with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();

    // Cache local pointers with restrict
    double* CFD_RESTRICT local_q[5];
    for (std::size_t v = 0; v < 5; ++v) {
        local_q[v] = q[v];
    }

    const double q_inlet[5] = {p.prs_inlet, p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // In supersonic inflow, all characteristics enter domain: full Dirichlet state
        for (std::size_t v = 0; v < 5; ++v) {
            apply_fixed_value_bc(local_q[v][gh], local_q[v][in], q_inlet[v]);
        }

        // Numerical guards for positive thermodynamic state
        local_q[0][gh] = std::max(local_q[0][gh], 1.0);
        local_q[4][gh] = std::max(local_q[4][gh], 1.0);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Supersonic Inlet */
inline void supersonic_inlet_grad_kernel(std::span<const double* const> q, 
                                         std::span<double* const> gx, 
                                         std::span<double* const> gy, 
                                         std::span<double* const> gz, 
                                         const mesh::MeshPart& m,
                                         const LocalIndex fbeg, 
                                         const LocalIndex fend, 
                                         const SupersonicInletParams& p) noexcept {
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

    const double q_inlet[5] = {p.prs_inlet, p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

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

        // Fixed value gradients for all 5 primitive variables
        for (std::size_t v = 0; v < 5; ++v) {
            apply_grad_fixed_value_bc(gx[v][gh], gy[v][gh], gz[v][gh],
                                      gx[v][in], gy[v][in], gz[v][in],
                                      q[v][in], q_inlet[v],
                                      nx, ny, nz, rcfn_inv);
        }

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class SupersonicInletBC
 * @brief Supersonic Inlet boundary condition with fully specified Dirichlet state.
 */
template <eos::EquationOfStatePolicy EOS>
class SupersonicInletBC final : public BoundaryCondition<EOS> {
public:
    SupersonicInletBC(std::string zone,
                      const LocalIndex fbeg,
                      const LocalIndex fend,
                      const SupersonicInletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "SupersonicInletBC requires exactly 5 mean-flow variables");
        supersonic_inlet_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        supersonic_inlet_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SupersonicInlet; }

private:
    SupersonicInletParams m_p;
};

} // namespace cfd::solver::bc
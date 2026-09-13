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
#include "cfd/numerics/boundary/boundary_gradients.hpp"
#include "cfd/numerics/boundary/boundary_values.hpp"
#include "cfd/solver/compressible/bc/bc.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"

namespace cfd::solver::compressible::bc {

/** 
 * @brief Canonical resolved primitive state for Subsonic Inlet (fixed velocity & temperature, extrapolated pressure).
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
    template <eos::EquationOfStatePolicy EOS>
    static SubsonicInletParams from_mach_angles(const EOS& eos,
                                                const double T,
                                                const double mach,
                                                const double alpha_deg,
                                                const double beta_deg) noexcept {
        constexpr double kDegToRad = M_PI / 180.0;
        constexpr double kRefPressure = 101325.0;
        const double alpha_rad = alpha_deg * kDegToRad;
        const double beta_rad  = beta_deg  * kDegToRad;

        const double rho = eos.density_Tp(T, kRefPressure);
        const double a   = eos.sound_speed_rhop(rho, kRefPressure);
        const double v_mag = mach * a;

        const double u = v_mag * std::cos(alpha_rad) * std::cos(beta_rad);
        const double v = -v_mag * std::sin(beta_rad);
        const double w = v_mag * std::sin(alpha_rad) * std::cos(beta_rad);

        return SubsonicInletParams{u, v, w, T};
    }

    // 3. Mach number + unit direction vector (dx, dy, dz) + temperature T
    template <eos::EquationOfStatePolicy EOS>
    static SubsonicInletParams from_mach_direction(const EOS& eos,
                                                   const double T,
                                                   const double mach,
                                                   const double dx,
                                                   const double dy,
                                                   const double dz) noexcept {
        constexpr double kRefPressure = 101325.0;
        const double rho = eos.density_Tp(T, kRefPressure);
        const double a   = eos.sound_speed_rhop(rho, kRefPressure);
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

namespace {

/** @brief Fills ghost cells with state values for Subsonic Inlet */
inline void subsonic_inlet_kernel(std::span<double* const> q,
                                  const mesh::MeshPart& m,
                                  const LocalIndex fbeg,
                                  const LocalIndex fend,
                                  const SubsonicInletParams& p) noexcept {
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

    // Cache inlet parameters in registers
    const double vx_inlet  = p.vx_inlet;
    const double vy_inlet  = p.vy_inlet;
    const double vz_inlet  = p.vz_inlet;
    const double tmp_inlet = p.tmp_inlet;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // 1. Extrapolate pressure from interior: dp/dn = 0
        numerics::boundary::apply_extrapolate_value(prs[gh], prs[in]);

        // 2. Fixed Dirichlet velocities: v_ghost = 2 * v_inlet - v_in
        numerics::boundary::apply_dirichlet_value(vx[gh], vx[in], vx_inlet);
        numerics::boundary::apply_dirichlet_value(vy[gh], vy[in], vy_inlet);
        numerics::boundary::apply_dirichlet_value(vz[gh], vz[in], vz_inlet);

        // 3. Fixed Dirichlet temperature: T_ghost = 2 * T_inlet - T_in
        numerics::boundary::apply_dirichlet_value(tmp[gh], tmp[in], tmp_inlet);

        // Physical lower bound guard against negative temperature
        tmp[gh] = std::max(tmp[gh], 1.0);

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Subsonic Inlet */
inline void subsonic_inlet_grad_kernel(std::span<const double* const> q,
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& m,
                                       const LocalIndex fbeg,
                                       const LocalIndex fend,
                                       const SubsonicInletParams& p) noexcept {
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

    // Inlet target Dirichlet values: [vx, vy, vz, tmp]
    const double q_inlet[4] = {p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

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

        // 1. Extrapolate pressure gradient: dp/dn = 0
        numerics::boundary::apply_extrapolate_gradient(gx[0][gh], gy[0][gh], gz[0][gh],
                                                       gx[0][in], gy[0][in], gz[0][in]);

        // 2. Fixed value gradients for velocities (1..3) and temperature (4)
        for (std::size_t d = 0; d < 4; ++d) {
            const std::size_t v = 1 + d;
            numerics::boundary::apply_dirichlet_gradient(gx[v][gh], gy[v][gh], gz[v][gh],
                                                         gx[v][in], gy[v][in], gz[v][in],
                                                         q[v][in], q_inlet[d], nx, ny, nz, rcfn_inv);
        }

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class SubsonicInletBC
 * @brief Subsonic Inlet boundary condition implementation.
 */
template <eos::EquationOfStatePolicy EOS>
class SubsonicInletBC final : public BoundaryCondition<EOS> {
public:
    SubsonicInletBC(std::string zone,
                    const LocalIndex fbeg,
                    const LocalIndex fend,
                    const SubsonicInletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "SubsonicInletBC requires exactly 5 mean-flow variables");
        subsonic_inlet_kernel(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        subsonic_inlet_grad_kernel(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicInlet; }

private:
    SubsonicInletParams m_p;
};

} // namespace cfd::solver::bc
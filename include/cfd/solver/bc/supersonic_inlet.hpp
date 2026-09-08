#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/mesh/localmesh.hpp"

#include "cfd/solver/bc/bc_fill_values.hpp"
#include "cfd/solver/bc/bc_fill_gradients.hpp"


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

    // if pressure, velocity vector and temperature are given
    static SupersonicInletParams from_velocities(const double p,
                                                 const double u,
                                                 const double v,
                                                 const double w,
                                                 const double T) noexcept {
        return SupersonicInletParams{p, u, v, w, T};
    }

    // if pressure, mach, direction unit (or not unit) vector and temperature are given
    template <eos::EquationOfState EOS>
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

    // if pressure, mach, angel of attac, slip angel and temperature are given
    template <eos::EquationOfState EOS>
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

/** @brief Set value in ghost cell */
inline void supersonic_inlet_kernel(fields::PrimitiveView<double> s, 
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

    // Cache inlet parameters in registers
    const double prs_inlet = p.prs_inlet;
    const double vx_inlet  = p.vx_inlet;
    const double vy_inlet  = p.vy_inlet;
    const double vz_inlet  = p.vz_inlet;
    const double tmp_inlet = p.tmp_inlet;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                // ghost cell = n_cells + local boundary face idx

        // ==== Fixed values for all variables ====
        // ==== pressure ====
        apply_fixed_value_bc(s.prs[gh], s.prs[in], prs_inlet);
        // ==== x-velocity ====
        apply_fixed_value_bc(s.vx[gh], s.vx[in], vx_inlet);
        // ==== y-velocity ====
        apply_fixed_value_bc(s.vy[gh], s.vy[in], vy_inlet);
        // ==== z-velocity ====
        apply_fixed_value_bc(s.vz[gh], s.vz[in], vz_inlet);
        // ==== temperature ====
        apply_fixed_value_bc(s.tmp[gh], s.tmp[in], tmp_inlet);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell */
inline void supersonic_inlet_grad_kernel(fields::PrimitiveView<const double> s, 
                                         fields::PrimitiveGradView<double> s_grad, 
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

    // Cache inlet parameters in registers
    const double prs_inlet = p.prs_inlet;
    const double vx_inlet  = p.vx_inlet;
    const double vy_inlet  = p.vy_inlet;
    const double vz_inlet  = p.vz_inlet;
    const double tmp_inlet = p.tmp_inlet;

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                // ghost cell = n_cells + local boundary face idx

        // ==== unit normal ====
        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        // ==== face center ====
        const double fcx = fcx_ptr[face_idx];
        const double fcy = fcy_ptr[face_idx];
        const double fcz = fcz_ptr[face_idx];

        // ==== inner cell center ====
        const double ccx = ccx_ptr[in];
        const double ccy = ccy_ptr[in];
        const double ccz = ccz_ptr[in];

        // ==== vector from cell center to face center ====
        const double rcfx = fcx - ccx;
        const double rcfy = fcy - ccy;
        const double rcfz = fcz - ccz;
        
        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
        const double rcfn_inv = 1.0 / rcfn;
        double gx_in, gy_in, gz_in;

        // ==== fill gradients in ghost cell ====
        // ==== pressure ====
        gx_in = s_grad.dprs_dx(in); gy_in = s_grad.dprs_dy(in); gz_in = s_grad.dprs_dz(in);
        apply_grad_fixed_value_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                  gx_in, gy_in, gz_in, s.prs[in], prs_inlet,
                                  nx, ny, nz, rcfn_inv);

        // ==== x-velocity ====
        gx_in = s_grad.dvx_dx(in); gy_in = s_grad.dvx_dy(in); gz_in = s_grad.dvx_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                  gx_in, gy_in, gz_in, s.vx[in], vx_inlet,
                                  nx, ny, nz, rcfn_inv);

        // ==== y-velocity ====
        gx_in = s_grad.dvy_dx(in); gy_in = s_grad.dvy_dy(in); gz_in = s_grad.dvy_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                  gx_in, gy_in, gz_in, s.vy[in], vy_inlet,
                                  nx, ny, nz, rcfn_inv);
        
        // ==== z-velocity ====
        gx_in = s_grad.dvz_dx(in); gy_in = s_grad.dvz_dy(in); gz_in = s_grad.dvz_dz(in);
        apply_grad_fixed_value_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                  gx_in, gy_in, gz_in, s.vz[in], vz_inlet, 
                                  nx, ny, nz, rcfn_inv);

        // ==== temperature ====
        gx_in = s_grad.dtmp_dx(in); gy_in = s_grad.dtmp_dy(in); gz_in = s_grad.dtmp_dz(in);
        apply_grad_fixed_value_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                  gx_in, gy_in, gz_in, s.tmp[in], tmp_inlet, 
                                  nx, ny, nz, rcfn_inv);

        ++f_loc;
    }
}


/**
 * @class SupersonicInletBC
 * @brief Supersonic Inlet BC with fixed primitive variables.
 */
template <eos::EquationOfState EOS>
class SupersonicInletBC final : public BoundaryCondition<EOS> {
public:
    SupersonicInletBC(std::string zone,
                      const LocalIndex fbeg,
                      const LocalIndex fend,
                      const SupersonicInletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend), m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        supersonic_inlet_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        supersonic_inlet_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

[[nodiscard]] BCType kind() const noexcept override { return BCType::SupersonicInlet; }

private:
    SupersonicInletParams m_p;
};

} //namespace cfd

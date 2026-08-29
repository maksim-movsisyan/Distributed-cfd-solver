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
 * @brief Parameters for Subsonic Outlet (backpressure).
 */
struct SubsonicOutletParams {
    double prs_outlet{101325.0}; ///< Target static backpressure p_back [Pa]

    static SubsonicOutletParams from_pressure(const double p_back) noexcept {
        return SubsonicOutletParams{p_back};
    }
};

/** @brief Set value in ghost cell for Subsonic Outlet */
inline void subsonic_outlet_kernel(fields::PrimitiveView<double> s,
                                   const mesh::MeshPart& m,
                                   const LocalIndex fbeg,
                                   const LocalIndex fend,
                                   const SubsonicOutletParams& p) noexcept {
    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const auto in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const auto gh = n_cells + f_loc;                                  // ghost cell

        // ==== 1. Fixed Dirichlet static backpressure: p_face = p_outlet ====
        apply_fixed_value_bc(s.prs[gh], s.prs[in], p.prs_outlet);

        // ==== 2. Extrapolate velocity components from interior (Neumann: dv/dn = 0) ====
        apply_extrapolation0_bc(s.vx[gh], s.vx[in]);
        apply_extrapolation0_bc(s.vy[gh], s.vy[in]);
        apply_extrapolation0_bc(s.vz[gh], s.vz[in]);

        // ==== 3. Extrapolate temperature from interior (Neumann: dT/dn = 0) ====
        apply_extrapolation0_bc(s.tmp[gh], s.tmp[in]);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell for Subsonic Outlet */
inline void subsonic_outlet_grad_kernel(fields::ConstPrimitiveView s,
                                        fields::PrimitiveGradView<double> s_grad,
                                        const mesh::MeshPart& m,
                                        const LocalIndex fbeg,
                                        const LocalIndex fend,
                                        const SubsonicOutletParams& p) noexcept {
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

        // ==== 1. Fixed value pressure gradient ====
        const double gx_in = s_grad.dprs_dx(in);
        const double gy_in = s_grad.dprs_dy(in);
        const double gz_in = s_grad.dprs_dz(in);

        apply_grad_fixed_value_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                  gx_in, gy_in, gz_in, s.prs[in], p.prs_outlet,
                                  nx, ny, nz, rcfn_inv);

        // ==== 2. Extrapolate velocity gradients ====
        apply_grad_extrapolation0_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                     s_grad.dvx_dx(in), s_grad.dvx_dy(in), s_grad.dvx_dz(in));

        apply_grad_extrapolation0_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                     s_grad.dvy_dx(in), s_grad.dvy_dy(in), s_grad.dvy_dz(in));

        apply_grad_extrapolation0_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                     s_grad.dvz_dx(in), s_grad.dvz_dy(in), s_grad.dvz_dz(in));

        // ==== 3. Extrapolate temperature gradient ====
        apply_grad_extrapolation0_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                     s_grad.dtmp_dx(in), s_grad.dtmp_dy(in), s_grad.dtmp_dz(in));

        ++f_loc;
    }
}

/**
 * @class SubsonicOutletBC
 * @brief Subsonic Outlet boundary condition with imposed static backpressure.
 */
template <eos::EquationOfState EOS>
class SubsonicOutletBC final : public BoundaryCondition<EOS> {
public:
    SubsonicOutletBC(std::string zone,
                     const LocalIndex fbeg,
                     const LocalIndex fend,
                     const SubsonicOutletParams& p)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend),
          m_p(p) {}

    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh,
               const EOS& /*eos*/) const override {
        subsonic_outlet_kernel(state, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_grad(fields::ConstPrimitiveView state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        subsonic_outlet_grad_kernel(state, state_grad, mesh, this->m_begin, this->m_end, m_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicOutlet; }

private:
    SubsonicOutletParams m_p;
};

} // namespace cfd::solver::bc
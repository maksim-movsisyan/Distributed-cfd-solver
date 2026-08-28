#pragma once

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

/** @brief Set value in ghost cell */
inline void symmetry_kernel(fields::PrimitiveView<double> s, const mesh::MeshPart& m,
                            LocalIndex fbeg, LocalIndex fend) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                  // ghost cell = n_cells + local boundary face idx

        // ==== unit normal ====
        const double nx = m.face_normal_x[face_idx];
        const double ny = m.face_normal_y[face_idx];
        const double nz = m.face_normal_z[face_idx];

        // ==== Reflect velocity vector (Symmetry plane: u_n = 0) ====
        apply_slip_component_bc(s.vx[gh], s.vy[gh], s.vz[gh],
                                s.vx[in], s.vy[in], s.vz[in],
                                nx, ny, nz);

        // ==== Extrapolate scalars from inner cell (dp/dn = 0, dT/dn = 0) ====
        apply_extrapolation0_bc(s.prs[gh], s.prs[in]);
        apply_extrapolation0_bc(s.tmp[gh], s.tmp[in]);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell */
inline void symmetry_grad_kernel(fields::PrimitiveView<const double> s,
                                 fields::PrimitiveGradView<double> s_grad,
                                 const mesh::MeshPart& m,
                                 LocalIndex fbeg, LocalIndex fend) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                  // ghost cell = n_cells + local boundary face idx

        // ==== unit normal ====
        const double nx = m.face_normal_x[face_idx];
        const double ny = m.face_normal_y[face_idx];
        const double nz = m.face_normal_z[face_idx];

        // ==== face center ====
        const double fcx = m.face_centroid_x[face_idx];
        const double fcy = m.face_centroid_y[face_idx];
        const double fcz = m.face_centroid_z[face_idx];

        // ==== inner cell center ====
        const double ccx = m.cell_centroid_x[in];
        const double ccy = m.cell_centroid_y[in];
        const double ccz = m.cell_centroid_z[in];

        // ==== vector from cell center to face center ====
        const double rcfx = fcx - ccx;
        const double rcfy = fcy - ccy;
        const double rcfz = fcz - ccz;

        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
        const double rcfn_inv = 1.0 / rcfn;
        double gx_in, gy_in, gz_in;

        // ==== pressure grad (zero normal gradient: dp/dn = 0) ====
        gx_in = s_grad.dprs_dx(in); gy_in = s_grad.dprs_dy(in); gz_in = s_grad.dprs_dz(in);
        apply_grad_fixed_gradient_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                     gx_in, gy_in, gz_in, 0.0, nx, ny, nz);

        // ==== x-velocity grad ====
        gx_in = s_grad.dvx_dx(in); gy_in = s_grad.dvx_dy(in); gz_in = s_grad.dvx_dz(in);
        apply_grad_slip_component_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                     gx_in, gy_in, gz_in, s.vx[in], s.vx[gh], nx, ny, nz, rcfn_inv);

        // ==== y-velocity grad ====
        gx_in = s_grad.dvy_dx(in); gy_in = s_grad.dvy_dy(in); gz_in = s_grad.dvy_dz(in);
        apply_grad_slip_component_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                     gx_in, gy_in, gz_in, s.vy[in], s.vy[gh], nx, ny, nz, rcfn_inv);

        // ==== z-velocity grad ====
        gx_in = s_grad.dvz_dx(in); gy_in = s_grad.dvz_dy(in); gz_in = s_grad.dvz_dz(in);
        apply_grad_slip_component_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                     gx_in, gy_in, gz_in, s.vz[in], s.vz[gh], nx, ny, nz, rcfn_inv);

        // ==== temperature grad (zero normal gradient: dT/dn = 0) ====
        gx_in = s_grad.dtmp_dx(in); gy_in = s_grad.dtmp_dy(in); gz_in = s_grad.dtmp_dz(in);
        apply_grad_fixed_gradient_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                     gx_in, gy_in, gz_in, 0.0, nx, ny, nz);

        ++f_loc;
    }
}

/**
 * @class SymmetryBC
 * @brief Symmetry boundary condition implementation.
 */
class SymmetryBC : public BoundaryCondition {
public:
    /** @brief SymmetryBC constructor */
    SymmetryBC(std::string zone, LocalIndex fbeg, LocalIndex fend)
        : BoundaryCondition(std::move(zone), fbeg, fend) {}

    /** @brief SymmetryBC apply implementation */
    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh) const override {
        symmetry_kernel(state, mesh, m_begin, m_end);
    }

    /** @brief SymmetryBC apply gradient implementation */
    void apply_grad(fields::PrimitiveView<const double> state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        symmetry_grad_kernel(state, state_grad, mesh, m_begin, m_end);
    }

    BCType kind() const override { return BCType::Symmetry; }
};

} // namespace cfd::solver::bc
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
inline void supersonic_outlet_kernel(fields::PrimitiveView<double> s,
                                     const mesh::MeshPart& m,
                                     LocalIndex fbeg, LocalIndex fend) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                  // ghost cell = n_cells + local boundary face idx

        // ==== Extrapolation of all variables (zero-gradient Neumann) ====
        // ==== pressure ====
        apply_extrapolation0_bc(s.prs[gh], s.prs[in]);
        // ==== x-velocity ====
        apply_extrapolation0_bc(s.vx[gh], s.vx[in]);
        // ==== y-velocity ====
        apply_extrapolation0_bc(s.vy[gh], s.vy[in]);
        // ==== z-velocity ====
        apply_extrapolation0_bc(s.vz[gh], s.vz[in]);
        // ==== temperature ====
        apply_extrapolation0_bc(s.tmp[gh], s.tmp[in]);

        ++f_loc;
    }
}

/** @brief Set gradient in ghost cell */
inline void supersonic_outlet_grad_kernel(fields::PrimitiveGradView<double> s_grad,
                                          const mesh::MeshPart& m,
                                          LocalIndex fbeg, LocalIndex fend) {
    std::size_t n_cells = static_cast<std::size_t>(m.n_cells);
    std::size_t n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = static_cast<std::size_t>(fbeg) - n_inner_faces;

    for (std::size_t face_idx = static_cast<std::size_t>(fbeg);
         face_idx < static_cast<std::size_t>(fend); ++face_idx) {
        const std::size_t in = static_cast<std::size_t>(m.face_owner[face_idx]); // inner (real) cell
        const std::size_t gh = n_cells + f_loc;                                  // ghost cell = n_cells + local boundary face idx

        // ==== fill gradients in ghost cell ====
        // ==== pressure ====
        apply_grad_extrapolation0_bc(s_grad.dprs_dx(gh), s_grad.dprs_dy(gh), s_grad.dprs_dz(gh),
                                     s_grad.dprs_dx(in), s_grad.dprs_dy(in), s_grad.dprs_dz(in));
        // ==== x-velocity ====
        apply_grad_extrapolation0_bc(s_grad.dvx_dx(gh), s_grad.dvx_dy(gh), s_grad.dvx_dz(gh),
                                     s_grad.dvx_dx(in), s_grad.dvx_dy(in), s_grad.dvx_dz(in));
        // ==== y-velocity ====
        apply_grad_extrapolation0_bc(s_grad.dvy_dx(gh), s_grad.dvy_dy(gh), s_grad.dvy_dz(gh),
                                     s_grad.dvy_dx(in), s_grad.dvy_dy(in), s_grad.dvy_dz(in));
        // ==== z-velocity ====
        apply_grad_extrapolation0_bc(s_grad.dvz_dx(gh), s_grad.dvz_dy(gh), s_grad.dvz_dz(gh),
                                     s_grad.dvz_dx(in), s_grad.dvz_dy(in), s_grad.dvz_dz(in));
        // ==== temperature ====
        apply_grad_extrapolation0_bc(s_grad.dtmp_dx(gh), s_grad.dtmp_dy(gh), s_grad.dtmp_dz(gh),
                                     s_grad.dtmp_dx(in), s_grad.dtmp_dy(in), s_grad.dtmp_dz(in));

        ++f_loc;
    }
}

/**
 * @class SupersonicOutletBC
 * @brief Supersonic outlet boundary condition implementation.
 * All variables and gradients are extrapolated from the inner adjacent cells.
 */
class SupersonicOutletBC : public BoundaryCondition {
public:
    /** @brief SupersonicOutletBC constructor */
    SupersonicOutletBC(std::string zone, LocalIndex fbeg, LocalIndex fend)
        : BoundaryCondition(std::move(zone), fbeg, fend) {}

    /** @brief SupersonicOutletBC apply implementation */
    void apply(fields::PrimitiveView<double> state,
               const mesh::MeshPart& mesh) const override {
        supersonic_outlet_kernel(state, mesh, m_begin, m_end);
    }

    /** @brief SupersonicOutletBC apply gradient implementation */
    void apply_grad(fields::PrimitiveView<const double> state,
                    fields::PrimitiveGradView<double> state_grad,
                    const mesh::MeshPart& mesh) const override {
        (void)state;
        supersonic_outlet_grad_kernel(state_grad, mesh, m_begin, m_end);
    }

    BCType kind() const override { return BCType::SupersonicOutlet; }
};

} // namespace cfd::solver::bc
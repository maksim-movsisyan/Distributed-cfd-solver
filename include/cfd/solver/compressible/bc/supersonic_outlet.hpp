#pragma once

#include <cassert>
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

namespace {

/** @brief Fills ghost cells with state values for Supersonic Outlet */
inline void supersonic_outlet_kernel(std::span<double* const> q,
                                     const mesh::MeshPart& m,
                                     const LocalIndex fbeg,
                                     const LocalIndex fend) noexcept {
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

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // In supersonic outflow, all characteristics exit domain: full Neumann extrapolation
        for (std::size_t v = 0; v < 5; ++v) {
            numerics::boundary::apply_extrapolate_value(local_q[v][gh], local_q[v][in]);
        }

        ++f_loc;
    }
}

/** @brief Fills ghost cells with gradients for Supersonic Outlet */
inline void supersonic_outlet_grad_kernel(std::span<double* const> gx,
                                          std::span<double* const> gy,
                                          std::span<double* const> gz,
                                          const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    // Unpack topology array with restrict
    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        // Zero-order extrapolation of all gradient components
        for (std::size_t v = 0; v < 5; ++v) {
            numerics::boundary::apply_extrapolate_gradient(gx[v][gh], gy[v][gh], gz[v][gh],
                                                           gx[v][in], gy[v][in], gz[v][in]);
        }

        ++f_loc;
    }
}

} // anonymous namespace

/**
 * @class SupersonicOutletBC
 * @brief Supersonic outlet boundary condition implementation.
 * All variables and gradients are fully extrapolated from the adjacent interior cells.
 */
template <eos::EquationOfStatePolicy EOS>
class SupersonicOutletBC final : public BoundaryCondition<EOS> {
public:
    SupersonicOutletBC(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : BoundaryCondition<EOS>(std::move(zone), fbeg, fend) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh,
                            const EOS& /*eos*/) const override {
        assert(q.size() == 5 && "SupersonicOutletBC requires exactly 5 mean-flow variables");
        supersonic_outlet_kernel(q, mesh, this->m_begin, this->m_end);
    }

    void update_ghost_cells_grad(std::span<const double* const> /*q*/,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        supersonic_outlet_grad_kernel(gx, gy, gz, mesh, this->m_begin, this->m_end);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SupersonicOutlet; }
};

} // namespace cfd::solver::bc
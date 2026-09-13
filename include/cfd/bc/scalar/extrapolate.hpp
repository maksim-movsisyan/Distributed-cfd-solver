#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/numerics/boundary/boundary_gradients.hpp"
#include "cfd/numerics/boundary/boundary_traversal.hpp"
#include "cfd/numerics/boundary/boundary_values.hpp"
#include "cfd/bc/scalar/scalar_bc.hpp"

namespace cfd::bc::scalar {

class ExtrapolateBC final : public ScalarBoundaryCondition {
public:
    ExtrapolateBC(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : ScalarBoundaryCondition(std::move(zone), fbeg, fend) {}

    void update_ghost_cells(std::span<double*> phi,
                            const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();

        // Fast path: pure zero-order extrapolation does not require normals or metrics
        numerics::boundary::for_each_boundary_face(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    numerics::boundary::apply_extrapolate_value(phi[var][gh], phi[var][in]);
                }
            });
    }

    void update_ghost_cells_grad(std::span<const double*> /*phi*/,
                                 std::span<double*> gx,
                                 std::span<double*> gy,
                                 std::span<double*> gz,
                                 const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = gx.size();
        assert(n_vars == gy.size() && n_vars == gz.size());

        // Fast path: gradient extrapolation copies values directly without metric projections
        numerics::boundary::for_each_boundary_face(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    numerics::boundary::apply_extrapolate_gradient(
                        gx[var][gh], gy[var][gh], gz[var][gh],
                        gx[var][in], gy[var][in], gz[var][in]);
                }
            });
    }

    [[nodiscard]] ScalarBCType kind() const noexcept override { return ScalarBCType::Extrapolate; }
};

} // namespace cfd::bc::scalar
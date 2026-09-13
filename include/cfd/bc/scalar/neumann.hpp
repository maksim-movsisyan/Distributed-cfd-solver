#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/numerics/boundary/boundary_gradients.hpp"
#include "cfd/numerics/boundary/boundary_traversal.hpp"
#include "cfd/numerics/boundary/boundary_values.hpp"
#include "cfd/bc/scalar/scalar_bc.hpp"

namespace cfd::bc::scalar {

class NeumannBC final : public ScalarBoundaryCondition {
public:
    NeumannBC(std::string zone,
              const LocalIndex fbeg, const LocalIndex fend,
              std::vector<double> gradients = {0.0})
        : ScalarBoundaryCondition(std::move(zone), fbeg, fend), m_gradients(std::move(gradients)) {}

    void update_ghost_cells(std::span<double*> phi,
                            const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();
        assert(n_vars == m_gradients.size() && "Mismatch between span size and BC gradients count");

        const double* CFD_RESTRICT grads = m_gradients.data();

        // Check if all gradients are zero (adiabatic / symmetry / zero-flux condition)
        const bool all_zero = std::all_of(m_gradients.begin(), m_gradients.end(), [](const double g) noexcept {
            return std::abs(g) < 1.0e-15;
        });

        if (all_zero) {
            // Fast path: does not read normals, centroids, or calculate projection metrics
            numerics::boundary::for_each_boundary_face(mesh, this->m_begin, this->m_end,
                [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
                    for (std::size_t var = 0; var < n_vars; ++var) {
                        numerics::boundary::apply_extrapolate_value(phi[var][gh], phi[var][in]);
                    }
                });
        } else {
            // Non-zero gradient path: projection distance rcfn is calculated once per face for all variables
            numerics::boundary::for_each_boundary_face_grad(mesh, this->m_begin, this->m_end,
                [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                    double /*nx*/, double /*ny*/, double /*nz*/, double rcfn_inv) noexcept {
                    const double rcfn = 1.0 / rcfn_inv;
                    for (std::size_t var = 0; var < n_vars; ++var) {
                        numerics::boundary::apply_neumann_value(phi[var][gh], phi[var][in], grads[var], rcfn);
                    }
                });
        }
    }

    void update_ghost_cells_grad(std::span<const double*> /*phi*/,
                                 std::span<double*> gx,
                                 std::span<double*> gy,
                                 std::span<double*> gz,
                                 const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = gx.size();
        assert(n_vars == m_gradients.size() && "Mismatch between span size and BC gradients count");
        assert(n_vars == gy.size() && n_vars == gz.size());

        const double* CFD_RESTRICT grads = m_gradients.data();

        // Neumann gradient only requires unit face normals, skipping centroids
        numerics::boundary::for_each_boundary_face_normal(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                double nx, double ny, double nz) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    numerics::boundary::apply_neumann_gradient(
                        gx[var][gh], gy[var][gh], gz[var][gh],
                        gx[var][in], gy[var][in], gz[var][in],
                        grads[var], nx, ny, nz);
                }
            });
    }

    [[nodiscard]] ScalarBCType kind() const noexcept override { return ScalarBCType::Neumann; }

    [[nodiscard]] const std::vector<double>& gradients() const noexcept { return m_gradients; }
    void set_gradients(std::vector<double> grad) noexcept { m_gradients = std::move(grad); }
    void add_gradient(const double grad) { m_gradients.push_back(grad); }

private:
    std::vector<double> m_gradients;
};

} // namespace cfd::bc::scalar
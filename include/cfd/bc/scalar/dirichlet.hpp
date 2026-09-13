#pragma once

#include <cassert>
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

class DirichletBC final : public ScalarBoundaryCondition {
public:
    DirichletBC(std::string zone, 
                const LocalIndex fbeg, const LocalIndex fend, 
                std::vector<double> values)
        : ScalarBoundaryCondition(std::move(zone), fbeg, fend),  m_values(std::move(values)) {}

    void update_ghost_cells(std::span<double*> phi,
                            const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();
        assert(n_vars == m_values.size() && "Mismatch between span size and BC values count");

        const double* CFD_RESTRICT vals = m_values.data();

        numerics::boundary::for_each_boundary_face(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    numerics::boundary::apply_dirichlet_value(phi[var][gh], phi[var][in], vals[var]);
                }
            });
    }

    void update_ghost_cells_grad(std::span<const double*> phi,
                                 std::span<double*> gx,
                                 std::span<double*> gy,
                                 std::span<double*> gz,
                                 const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();
        assert(n_vars == m_values.size() && "Mismatch between span size and BC values count");
        assert(n_vars == gx.size() && n_vars == gy.size() && n_vars == gz.size());

        const double* CFD_RESTRICT vals = m_values.data();

        numerics::boundary::for_each_boundary_face_grad(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                double nx, double ny, double nz, double rcfn_inv) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    numerics::boundary::apply_dirichlet_gradient(
                        gx[var][gh], gy[var][gh], gz[var][gh],
                        gx[var][in], gy[var][in], gz[var][in],
                        phi[var][in], vals[var],
                        nx, ny, nz, rcfn_inv);
                }
            });
    }

    [[nodiscard]] ScalarBCType kind() const noexcept override { return ScalarBCType::Dirichlet; }
    
    [[nodiscard]] const std::vector<double>& values() const noexcept { return m_values; }
    void set_values(std::vector<double> val) noexcept { m_values = std::move(val); }
    void add_value(const double val) { m_values.push_back(val); }

private:
    std::vector<double> m_values;
};

} // namespace cfd::bc::scalar
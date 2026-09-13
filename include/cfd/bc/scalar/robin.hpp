#pragma once

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
#include "cfd/bc/scalar/scalar_bc.hpp"

namespace cfd::bc::scalar {

struct RobinParams {
    double a{1.0}; ///< Coefficient for value term: a * phi
    double b{0.0}; ///< Coefficient for flux term:  b * (dphi/dn)
    double c{0.0}; ///< Target right hand side:     a * phi + b * (dphi/dn) = c
};

class RobinBC final : public ScalarBoundaryCondition {
public:
    RobinBC(std::string zone,
            const LocalIndex fbeg,
            const LocalIndex fend,
            std::vector<RobinParams> params = {{1.0, 0.0, 0.0}})
        : ScalarBoundaryCondition(std::move(zone), fbeg, fend), m_params(std::move(params)) {}

    RobinBC(std::string zone,
            const LocalIndex fbeg,
            const LocalIndex fend,
            const RobinParams& p)
        : ScalarBoundaryCondition(std::move(zone), fbeg, fend), m_params{p} {}

    void update_ghost_cells(std::span<double*> phi,
                            const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();
        assert(n_vars == m_params.size() && "Mismatch between span size and BC params count");

        const RobinParams* CFD_RESTRICT pars = m_params.data();

        numerics::boundary::for_each_boundary_face_grad(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                double /*nx*/, double /*ny*/, double /*nz*/, double rcfn_inv) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    const double a = pars[var].a;
                    const double b = pars[var].b;
                    const double c = pars[var].c;

                    const double b_over_rcfn = b * rcfn_inv;
                    const double denom = a + b_over_rcfn;
                    const double safe_denom = (std::abs(denom) > 1.0e-14) ? denom : 1.0e-14;

                    phi[var][gh] = (2.0 * c - phi[var][in] * (a - b_over_rcfn)) / safe_denom;
                }
            });
    }

    void update_ghost_cells_grad(std::span<const double*> phi,
                                 std::span<double*> gx,
                                 std::span<double*> gy,
                                 std::span<double*> gz,
                                 const mesh::MeshPart& mesh) const override {
        const std::size_t n_vars = phi.size();
        assert(n_vars == m_params.size() && "Mismatch between span size and BC params count");
        assert(n_vars == gx.size() && n_vars == gy.size() && n_vars == gz.size());

        const RobinParams* CFD_RESTRICT pars = m_params.data();

        numerics::boundary::for_each_boundary_face_grad(mesh, this->m_begin, this->m_end,
            [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
                double nx, double ny, double nz, double rcfn_inv) noexcept {
                for (std::size_t var = 0; var < n_vars; ++var) {
                    const double a = pars[var].a;
                    const double b = pars[var].b;
                    const double c = pars[var].c;

                    const double b_over_rcfn = b * rcfn_inv;
                    const double denom = a + b_over_rcfn;
                    const double safe_denom = (std::abs(denom) > 1.0e-14) ? denom : 1.0e-14;

                    const double phi_gh = (2.0 * c - phi[var][in] * (a - b_over_rcfn)) / safe_denom;
                    const double phi_face = 0.5 * (phi[var][in] + phi_gh);

                    numerics::boundary::apply_dirichlet_gradient(
                        gx[var][gh], gy[var][gh], gz[var][gh],
                        gx[var][in], gy[var][in], gz[var][in],
                        phi[var][in], phi_face,
                        nx, ny, nz, rcfn_inv);
                }
            });
    }

    [[nodiscard]] ScalarBCType kind() const noexcept override { return ScalarBCType::Robin; }

    [[nodiscard]] const std::vector<RobinParams>& params() const noexcept { return m_params; }
    void set_params(std::vector<RobinParams> p) noexcept { m_params = std::move(p); }
    void add_params(const RobinParams& p) { m_params.push_back(p); }

private:
    std::vector<RobinParams> m_params;
};

} // namespace cfd::bc::scalar
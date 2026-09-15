#pragma once

#include <algorithm>
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
#include "cfd/bc/physical/physical_bc.hpp"

namespace cfd::bc::physical {

/** 
 * @brief Parameters for Subsonic Outlet (imposed static backpressure).
 */
struct SubsonicOutletParams {
    double prs_outlet{101325.0}; ///< Target static backpressure p_back [Pa]
};

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for Subsonic Outlet.
 * Imposes Dirichlet pressure and extrapolates all remaining components.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void subsonic_outlet_kernel(std::span<double* const> q,
                                   const mesh::MeshPart& m,
                                   const LocalIndex fbeg,
                                   const LocalIndex fend,
                                   const SubsonicOutletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache primitive field pointers with restrict
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    const double prs_outlet = p.prs_outlet;

    // Values traversal: no normal or metric arrays needed
    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            // 1. Fixed Dirichlet static backpressure
            numerics::boundary::apply_dirichlet_value(lq[0][gh], lq[0][in], prs_outlet);
            lq[0][gh] = std::max(lq[0][gh], 1.0);

            // 2. Extrapolate velocity components (and temperature if NVars == 5)
            for (std::size_t v = 1; v < NVars; ++v) {
                numerics::boundary::apply_extrapolate_value(lq[v][gh], lq[v][in]);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Subsonic Outlet.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void subsonic_outlet_grad_kernel(std::span<const double* const> q,
                                        std::span<double* const> gx,
                                        std::span<double* const> gy,
                                        std::span<double* const> gz,
                                        const mesh::MeshPart& m,
                                        const LocalIndex fbeg,
                                        const LocalIndex fend,
                                        const SubsonicOutletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    const double* CFD_RESTRICT prs_in = q[0];

    double* CFD_RESTRICT lgx[NVars];
    double* CFD_RESTRICT lgy[NVars];
    double* CFD_RESTRICT lgz[NVars];

    for (std::size_t v = 0; v < NVars; ++v) {
        lgx[v] = gx[v];
        lgy[v] = gy[v];
        lgz[v] = gz[v];
    }

    const double prs_outlet = p.prs_outlet;

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // 1. Fixed value pressure gradient (Dirichlet)
            numerics::boundary::apply_dirichlet_gradient(
                lgx[0][gh], lgy[0][gh], lgz[0][gh],
                lgx[0][in], lgy[0][in], lgz[0][in],
                prs_in[in], prs_outlet,
                nx, ny, nz, rcfn_inv);

            // 2. Extrapolate velocity and temperature gradients (Neumann: d(grad)/dn = 0)
            for (std::size_t v = 1; v < NVars; ++v) {
                numerics::boundary::apply_extrapolate_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in]);
            }
        });
}

} // anonymous namespace

/**
 * @class SubsonicOutletBC
 * @brief Subsonic Outlet boundary condition with imposed static backpressure.
 * Unified for both 4-component (incompressible) and 5-component (compressible) solvers.
 */
class SubsonicOutletBC final : public BoundaryCondition {
public:
    SubsonicOutletBC(std::string zone,
                     const LocalIndex fbeg,
                     const LocalIndex fend,
                     const SubsonicOutletParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "SubsonicOutletBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::subsonic_outlet_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::subsonic_outlet_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && gx.size() >= 4 && gy.size() >= 4 && gz.size() >= 4);
        assert(q.size() == gx.size() && gx.size() == gy.size() && gy.size() == gz.size());

        if (q.size() >= 5) {
            kernels::subsonic_outlet_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::subsonic_outlet_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void apply_momentum_bc(const mesh::MeshPart& mesh,
                       const mesh::MeshAuxGeometry& aux_geom,
                       double* CFD_RESTRICT diag,
                       double* CFD_RESTRICT rhs_u,
                       double* CFD_RESTRICT rhs_v,
                       double* CFD_RESTRICT rhs_w,
                       double* CFD_RESTRICT m_dot,
                       const double rho,
                       const double mu,
                       const double* CFD_RESTRICT mut = nullptr) const override {
    static_cast<void>(aux_geom);
    static_cast<void>(rhs_u);
    static_cast<void>(rhs_v);
    static_cast<void>(rhs_w);
    static_cast<void>(rho);
    static_cast<void>(mu);
    static_cast<void>(mut);

    assert(m_dot != nullptr && "PressureOutlet requires valid mass flux field m_dot");

    for (LocalIndex face_idx = m_begin; face_idx < m_end; ++face_idx) {
        const auto f = static_cast<std::size_t>(face_idx);
        const auto owner = static_cast<std::size_t>(mesh.face_owner[f]);

        const double mdot_b = m_dot[f];

        const double mdot_pos = (mdot_b > 0.0) ? mdot_b : 0.0;

        diag[owner] += mdot_pos;
    }
}

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicOutlet; }

private:
    SubsonicOutletParams m_p;
};

} // namespace cfd::bc::physical
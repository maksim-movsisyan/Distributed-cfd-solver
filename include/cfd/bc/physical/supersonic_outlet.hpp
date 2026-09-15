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
#include "cfd/bc/physical/physical_bc.hpp"

namespace cfd::bc::physical {

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for Supersonic Outlet (full zero-order Neumann extrapolation).
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void supersonic_outlet_kernel(std::span<double* const> q,
                                     const mesh::MeshPart& m,
                                     const LocalIndex fbeg,
                                     const LocalIndex fend) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Pre-cache pointers in registers with restrict
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    // Fast traversal: only cell indices are needed, no normals or metrics loaded from RAM
    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            for (std::size_t v = 0; v < NVars; ++v) {
                numerics::boundary::apply_extrapolate_value(lq[v][gh], lq[v][in]);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Supersonic Outlet (full zero-order Neumann extrapolation).
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void supersonic_outlet_grad_kernel(std::span<double* const> gx,
                                          std::span<double* const> gy,
                                          std::span<double* const> gz,
                                          const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Pre-cache gradient pointers in registers
    double* CFD_RESTRICT lgx[NVars];
    double* CFD_RESTRICT lgy[NVars];
    double* CFD_RESTRICT lgz[NVars];

    for (std::size_t v = 0; v < NVars; ++v) {
        lgx[v] = gx[v];
        lgy[v] = gy[v];
        lgz[v] = gz[v];
    }

    // Fast traversal: zero-order extrapolation does not require face normals or centroids
    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            for (std::size_t v = 0; v < NVars; ++v) {
                numerics::boundary::apply_extrapolate_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in]);
            }
        });
}

} // namespace kernels

/**
 * @class SupersonicOutletBC
 * @brief Supersonic outlet boundary condition implementation.
 * All characteristics exit domain: full zero-order Neumann extrapolation for both state and gradients.
 */
class SupersonicOutletBC final : public BoundaryCondition {
public:
    SupersonicOutletBC(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : BoundaryCondition(std::move(zone), fbeg, fend) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "SupersonicOutletBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::supersonic_outlet_kernel<5>(q, mesh, this->m_begin, this->m_end);
        } else {
            kernels::supersonic_outlet_kernel<4>(q, mesh, this->m_begin, this->m_end);
        }
    }

    void update_ghost_cells_grad(std::span<const double* const> /*q*/,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(gx.size() >= 4 && gy.size() >= 4 && gz.size() >= 4);
        assert(gx.size() == gy.size() && gy.size() == gz.size());

        if (gx.size() >= 5) {
            kernels::supersonic_outlet_grad_kernel<5>(gx, gy, gz, mesh, this->m_begin, this->m_end);
        } else {
            kernels::supersonic_outlet_grad_kernel<4>(gx, gy, gz, mesh, this->m_begin, this->m_end);
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
        static_cast<void>(mesh);
        static_cast<void>(aux_geom);
        static_cast<void>(diag);
        static_cast<void>(rhs_u);
        static_cast<void>(rhs_v);
        static_cast<void>(rhs_w);
        static_cast<void>(m_dot);
        static_cast<void>(rho);
        static_cast<void>(mu);
        static_cast<void>(mut);
    }

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SupersonicOutlet; }
};

} // namespace cfd::bc::physical
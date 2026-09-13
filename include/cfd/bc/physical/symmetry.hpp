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

namespace {

/** 
 * @brief Fills ghost cells with state values for Symmetry plane boundary.
 * @tparam NVars Number of active variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void symmetry_kernel(std::span<double* const> q,
                            const mesh::MeshPart& m,
                            const LocalIndex fbeg,
                            const LocalIndex fend) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache pointers with restrict in registers: zero indirection overhead
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    numerics::boundary::for_each_boundary_face_normal(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz) noexcept {
            // 1. Reflect velocity vector across symmetry plane (u_n = 0)
            numerics::boundary::apply_slip_component(lq[1][gh], lq[2][gh], lq[3][gh],
                                                     lq[1][in], lq[2][in], lq[3][in],
                                                     nx, ny, nz);

            // 2. Extrapolate pressure across symmetry plane (dp/dn = 0)
            numerics::boundary::apply_extrapolate_value(lq[0][gh], lq[0][in]);

            // 3. Extrapolate temperature if present (dT/dn = 0)
            if constexpr (NVars >= 5) {
                numerics::boundary::apply_extrapolate_value(lq[4][gh], lq[4][in]);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Symmetry plane boundary.
 * @tparam NVars Number of active variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void symmetry_grad_kernel(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& m,
                                 const LocalIndex fbeg,
                                 const LocalIndex fend) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache all field and gradient pointers in registers
    const double* CFD_RESTRICT lq[NVars];
    double* CFD_RESTRICT lgx[NVars];
    double* CFD_RESTRICT lgy[NVars];
    double* CFD_RESTRICT lgz[NVars];

    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v]  = q[v];
        lgx[v] = gx[v];
        lgy[v] = gy[v];
        lgz[v] = gz[v];
    }

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // 1. Pressure gradient: zero normal gradient (dp/dn = 0)
            numerics::boundary::apply_neumann_gradient(lgx[0][gh], lgy[0][gh], lgz[0][gh],
                                                       lgx[0][in], lgy[0][in], lgz[0][in],
                                                       0.0, nx, ny, nz);

            // 2. Velocity gradients: slip reflection
            for (std::size_t d = 0; d < 3; ++d) {
                const std::size_t v = 1 + d;
                numerics::boundary::apply_slip_gradient(lgx[v][gh], lgy[v][gh], lgz[v][gh],
                                                        lgx[v][in], lgy[v][in], lgz[v][in],
                                                        lq[v][in], lq[v][gh], nx, ny, nz, rcfn_inv);
            }

            // 3. Temperature gradient if present: symmetry (dT/dn = 0)
            if constexpr (NVars >= 5) {
                numerics::boundary::apply_neumann_gradient(lgx[4][gh], lgy[4][gh], lgz[4][gh],
                                                           lgx[4][in], lgy[4][in], lgz[4][in],
                                                           0.0, nx, ny, nz);
            }
        });
}

} // anonymous namespace

/**
 * @class SymmetryBC
 * @brief Symmetry boundary condition implementation for planar reflection.
 * Unified for compressible (5 vars) and incompressible (4 vars) solvers.
 */
class SymmetryBC final : public BoundaryCondition {
public:
    SymmetryBC(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : BoundaryCondition(std::move(zone), fbeg, fend) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "SymmetryBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            symmetry_kernel<5>(q, mesh, this->m_begin, this->m_end);
        } else {
            symmetry_kernel<4>(q, mesh, this->m_begin, this->m_end);
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
            symmetry_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end);
        } else {
            symmetry_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end);
        }
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::Symmetry; }
};

} // namespace cfd::bc::physical
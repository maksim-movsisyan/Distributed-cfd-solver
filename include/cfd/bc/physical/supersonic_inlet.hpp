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
 * @brief Canonical resolved primitive state for Supersonic Inlet.
 * Direct physical quantities without EOS coupling.
 */
struct SupersonicInletParams {
    double prs_inlet{101325.0}; ///< Static pressure [Pa]
    double vx_inlet{0.0};        ///< Velocity X [m/s]
    double vy_inlet{0.0};        ///< Velocity Y [m/s]
    double vz_inlet{0.0};        ///< Velocity Z [m/s]
    double tmp_inlet{288.15};    ///< Static temperature [K]
};

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for Supersonic Inlet.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void supersonic_inlet_kernel(std::span<double* const> q, 
                                    const mesh::MeshPart& m, 
                                    const LocalIndex fbeg, 
                                    const LocalIndex fend, 
                                    const SupersonicInletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache pointers in registers: eliminates span dereference inside face loop
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    const double q_inlet[5] = {p.prs_inlet, p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            // Unrolled loop over all active variables (pure Dirichlet)
            for (std::size_t v = 0; v < NVars; ++v) {
                numerics::boundary::apply_dirichlet_value(lq[v][gh], lq[v][in], q_inlet[v]);
            }

            // Numerical safety bounds for thermodynamic variables
            lq[0][gh] = std::max(lq[0][gh], 1.0);
            if constexpr (NVars >= 5) {
                lq[4][gh] = std::max(lq[4][gh], 1.0);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Supersonic Inlet.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void supersonic_inlet_grad_kernel(std::span<const double* const> q, 
                                         std::span<double* const> gx, 
                                         std::span<double* const> gy, 
                                         std::span<double* const> gz, 
                                         const mesh::MeshPart& m, 
                                         const LocalIndex fbeg, 
                                         const LocalIndex fend, 
                                         const SupersonicInletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Pre-cache pointers in registers
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

    const double q_inlet[5] = {p.prs_inlet, p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // Loop bound is a compile-time constant: fully unrolled by compiler
            for (std::size_t v = 0; v < NVars; ++v) {
                numerics::boundary::apply_dirichlet_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in],
                    lq[v][in], q_inlet[v],
                    nx, ny, nz, rcfn_inv);
            }
        });
}

} // anonymous namespace

/**
 * @class SupersonicInletBC
 * @brief Supersonic Inlet boundary condition with fully specified Dirichlet state.
 * Unified for both 4-component and 5-component flow vectors.
 */
class SupersonicInletBC final : public BoundaryCondition {
public:
    SupersonicInletBC(std::string zone,
                      const LocalIndex fbeg,
                      const LocalIndex fend,
                      const SupersonicInletParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "SupersonicInletBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::supersonic_inlet_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::supersonic_inlet_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
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
            kernels::supersonic_inlet_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::supersonic_inlet_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SupersonicInlet; }

private:
    SupersonicInletParams m_p;
};

} // namespace cfd::bc::physical
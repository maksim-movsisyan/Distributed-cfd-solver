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
 * @brief Canonical resolved primitive state for Subsonic Inlet (fixed velocity and temperature, extrapolated pressure).
 * Direct physical quantities without EOS coupling.
 */
struct SubsonicInletParams {
    double vx_inlet{0.0};     ///< Prescribed velocity X [m/s]
    double vy_inlet{0.0};     ///< Prescribed velocity Y [m/s]
    double vz_inlet{0.0};     ///< Prescribed velocity Z [m/s]
    double tmp_inlet{288.15}; ///< Prescribed static temperature [K]
};

namespace kernels {

/** 
 * @brief Fills ghost cells with state values for Subsonic Inlet.
 * Pressure is extrapolated from the interior; velocity and temperature are Dirichlet.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void subsonic_inlet_kernel(std::span<double* const> q,
                                  const mesh::MeshPart& m,
                                  const LocalIndex fbeg,
                                  const LocalIndex fend,
                                  const SubsonicInletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache primitive field pointers with restrict
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    const double vx_in  = p.vx_inlet;
    const double vy_in  = p.vy_inlet;
    const double vz_in  = p.vz_inlet;
    const double tmp_in = p.tmp_inlet;

    // Values traversal: no normal or metric arrays needed
    numerics::boundary::for_each_boundary_face(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh) noexcept {
            // 1. Extrapolate pressure from interior (dp/dn = 0)
            numerics::boundary::apply_extrapolate_value(lq[0][gh], lq[0][in]);

            // 2. Fixed Dirichlet velocities
            numerics::boundary::apply_dirichlet_value(lq[1][gh], lq[1][in], vx_in);
            numerics::boundary::apply_dirichlet_value(lq[2][gh], lq[2][in], vy_in);
            numerics::boundary::apply_dirichlet_value(lq[3][gh], lq[3][in], vz_in);

            // 3. Fixed Dirichlet temperature if present
            if constexpr (NVars >= 5) {
                numerics::boundary::apply_dirichlet_value(lq[4][gh], lq[4][in], tmp_in);
                lq[4][gh] = std::max(lq[4][gh], 1.0);
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Subsonic Inlet.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void subsonic_inlet_grad_kernel(std::span<const double* const> q,
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& m,
                                       const LocalIndex fbeg,
                                       const LocalIndex fend,
                                       const SubsonicInletParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

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

    const double q_dirichlet[4] = {p.vx_inlet, p.vy_inlet, p.vz_inlet, p.tmp_inlet};

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            // 1. Extrapolate pressure gradient (dp/dn = 0)
            numerics::boundary::apply_extrapolate_gradient(
                lgx[0][gh], lgy[0][gh], lgz[0][gh],
                lgx[0][in], lgy[0][in], lgz[0][in]);

            // 2. Fixed value gradients for velocities (1..3) and temperature (4 if NVars == 5)
            for (std::size_t d = 0; d < NVars - 1; ++d) {
                const std::size_t v = 1 + d;
                numerics::boundary::apply_dirichlet_gradient(
                    lgx[v][gh], lgy[v][gh], lgz[v][gh],
                    lgx[v][in], lgy[v][in], lgz[v][in],
                    lq[v][in], q_dirichlet[d],
                    nx, ny, nz, rcfn_inv);
            }
        });
}

} // namespace kernels

/**
 * @class SubsonicInletBC
 * @brief Subsonic Inlet boundary condition implementation.
 * Unified for both 4-component (incompressible) and 5-component (compressible) solvers.
 */
class SubsonicInletBC final : public BoundaryCondition {
public:
    SubsonicInletBC(std::string zone,
                    const LocalIndex fbeg,
                    const LocalIndex fend,
                    const SubsonicInletParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "SubsonicInletBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::subsonic_inlet_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::subsonic_inlet_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
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
            kernels::subsonic_inlet_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::subsonic_inlet_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        }
    }

    void apply_momentum_bc(std::span<double*> diag_u,
                           std::span<double*> rhs,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs);
        static_cast<void>(diag_u);
    }

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] BCType kind() const noexcept override { return BCType::SubsonicInlet; }

private:
    SubsonicInletParams m_p;
};

} // namespace cfd::bc::physical
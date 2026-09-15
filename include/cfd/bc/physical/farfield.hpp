#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
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
 * @brief Precomputed freestream parameters for Characteristic Farfield BC.
 * Direct physical quantities: configuration loader computes Mach/angles beforehand.
 */
struct FarfieldParams {
    double prs_inf{101325.0};   ///< Static pressure p_inf [Pa]
    double tmp_inf{288.15};     ///< Static temperature T_inf [K]
    double vx_inf{0.0};         ///< Freestream velocity X [m/s]
    double vy_inf{0.0};         ///< Freestream velocity Y [m/s]
    double vz_inf{0.0};         ///< Freestream velocity Z [m/s]
    double gamma{1.4};          ///< Specific heat ratio [-]
    double R{287.052874};       ///< Specific gas constant [J / (kg K)]
};

namespace kernels {

/** 
 * @brief Helper evaluating boundary face state from 1D Riemann Invariants.
 */
inline void compute_riemann_farfield_state(const FarfieldParams& p,
                                           const double p_in, const double T_in,
                                           const double vx_in, const double vy_in, const double vz_in,
                                           const double nx, const double ny, const double nz,
                                           double& p_b, double& T_b,
                                           double& vx_b, double& vy_b, double& vz_b,
                                           bool& is_supersonic_outflow) noexcept {
    const double gm1 = p.gamma - 1.0;
    const double inv_gm1 = 1.0 / gm1;

    // Normal velocities
    const double un_in  = vx_in * nx + vy_in * ny + vz_in * nz;
    const double un_inf = p.vx_inf * nx + p.vy_inf * ny + p.vz_inf * nz;

    // Speeds of sound
    const double a_in  = std::sqrt(p.gamma * p.R * std::max(T_in, 1.0e-6));
    const double a_inf = std::sqrt(p.gamma * p.R * std::max(p.tmp_inf, 1.0e-6));

    const double mn_in = un_in / a_in;

    // 1. Supersonic Outflow: full extrapolation
    if (mn_in >= 1.0) {
        is_supersonic_outflow = true;
        p_b  = p_in;
        T_b  = T_in;
        vx_b = vx_in;
        vy_b = vy_in;
        vz_b = vz_in;
        return;
    }

    is_supersonic_outflow = false;

    // 2. Supersonic Inflow: full freestream Dirichlet
    if (mn_in <= -1.0) {
        p_b  = p.prs_inf;
        T_b  = p.tmp_inf;
        vx_b = p.vx_inf;
        vy_b = p.vy_inf;
        vz_b = p.vz_inf;
        return;
    }

    // 3 & 4. Subsonic Inflow / Outflow: Riemann Invariants
    const double R_out = un_in + 2.0 * a_in * inv_gm1;
    const double R_in  = un_inf - 2.0 * a_inf * inv_gm1;

    const double un_b = 0.5 * (R_out + R_in);
    double a_b = 0.25 * gm1 * (R_out - R_in);
    if (a_b <= 0.0) a_b = a_inf; // Safeguard against non-physical states

    double entropy_s = 0.0;

    if (un_b >= 0.0) {
        // Subsonic Outflow: tangential velocity and entropy from interior
        vx_b = vx_in + (un_b - un_in) * nx;
        vy_b = vy_in + (un_b - un_in) * ny;
        vz_b = vz_in + (un_b - un_in) * nz;

        const double rho_in = p_in / (p.R * std::max(T_in, 1.0e-6));
        entropy_s = p_in / std::pow(std::max(rho_in, 1.0e-12), p.gamma);
    } else {
        // Subsonic Inflow: tangential velocity and entropy from freestream
        vx_b = p.vx_inf + (un_b - un_inf) * nx;
        vy_b = p.vy_inf + (un_b - un_inf) * ny;
        vz_b = p.vz_inf + (un_b - un_inf) * nz;

        const double rho_inf = p.prs_inf / (p.R * std::max(p.tmp_inf, 1.0e-6));
        entropy_s = p.prs_inf / std::pow(std::max(rho_inf, 1.0e-12), p.gamma);
    }

    const double rho_b = std::pow((a_b * a_b) / (p.gamma * std::max(entropy_s, 1.0e-12)), inv_gm1);
    p_b = (rho_b * a_b * a_b) / p.gamma;
    T_b = (a_b * a_b) / (p.gamma * p.R);
}

/** 
 * @brief Fills ghost cells with state values for Characteristic Farfield.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void farfield_kernel(std::span<double* const> q,
                            const mesh::MeshPart& m,
                            const LocalIndex fbeg,
                            const LocalIndex fend,
                            const FarfieldParams& p) noexcept {
    static_assert(NVars == 4 || NVars == 5, "NVars must be 4 or 5");

    // Cache pointers with restrict in registers
    double* CFD_RESTRICT lq[NVars];
    for (std::size_t v = 0; v < NVars; ++v) {
        lq[v] = q[v];
    }

    numerics::boundary::for_each_boundary_face_normal(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz) noexcept {
            const double T_in = (NVars >= 5) ? lq[4][in] : p.tmp_inf;

            double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
            bool is_supersonic_outflow = false;

            compute_riemann_farfield_state(p, lq[0][in], T_in,
                                           lq[1][in], lq[2][in], lq[3][in],
                                           nx, ny, nz,
                                           pb, Tb, vxb, vyb, vzb,
                                           is_supersonic_outflow);

            if (is_supersonic_outflow) {
                for (std::size_t v = 0; v < NVars; ++v) {
                    numerics::boundary::apply_extrapolate_value(lq[v][gh], lq[v][in]);
                }
            } else {
                const double q_b[5] = {pb, vxb, vyb, vzb, Tb};

                for (std::size_t v = 0; v < NVars; ++v) {
                    numerics::boundary::apply_dirichlet_value(lq[v][gh], lq[v][in], q_b[v]);
                }

                // Numerical floor guards
                lq[0][gh] = std::max(lq[0][gh], 1.0);
                if constexpr (NVars >= 5) {
                    lq[4][gh] = std::max(lq[4][gh], 1.0);
                }
            }
        });
}

/** 
 * @brief Fills ghost cells with gradients for Characteristic Farfield.
 * @tparam NVars Number of variables (4 for Incompressible, 5 for Compressible).
 */
template <std::size_t NVars>
inline void farfield_grad_kernel(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& m,
                                 const LocalIndex fbeg,
                                 const LocalIndex fend,
                                 const FarfieldParams& p) noexcept {
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

    numerics::boundary::for_each_boundary_face_grad(m, fbeg, fend,
        [&](std::size_t /*face_idx*/, std::size_t in, std::size_t gh,
            double nx, double ny, double nz, double rcfn_inv) noexcept {
            const double T_in = (NVars >= 5) ? lq[4][in] : p.tmp_inf;

            double pb = 0.0, Tb = 0.0, vxb = 0.0, vyb = 0.0, vzb = 0.0;
            bool is_supersonic_outflow = false;

            compute_riemann_farfield_state(p, lq[0][in], T_in,
                                           lq[1][in], lq[2][in], lq[3][in],
                                           nx, ny, nz,
                                           pb, Tb, vxb, vyb, vzb,
                                           is_supersonic_outflow);

            if (is_supersonic_outflow) {
                // Outflow characteristics: pure gradient extrapolation
                for (std::size_t v = 0; v < NVars; ++v) {
                    numerics::boundary::apply_extrapolate_gradient(
                        lgx[v][gh], lgy[v][gh], lgz[v][gh],
                        lgx[v][in], lgy[v][in], lgz[v][in]);
                }
            } else {
                // Inflow/mixed characteristics: Dirichlet gradient to resolved boundary state
                const double q_b[5] = {pb, vxb, vyb, vzb, Tb};

                for (std::size_t v = 0; v < NVars; ++v) {
                    numerics::boundary::apply_dirichlet_gradient(
                        lgx[v][gh], lgy[v][gh], lgz[v][gh],
                        lgx[v][in], lgy[v][in], lgz[v][in],
                        lq[v][in], q_b[v],
                        nx, ny, nz, rcfn_inv);
                }
            }
        });
}

} // namespace kernels

/**
 * @class FarfieldBC
 * @brief Non-reflecting characteristic boundary condition based on 1D Riemann Invariants.
 * Unified for both 4-component and 5-component flow states.
 */
class FarfieldBC final : public BoundaryCondition {
public:
    FarfieldBC(std::string zone,
               const LocalIndex fbeg,
               const LocalIndex fend,
               const FarfieldParams& p)
        : BoundaryCondition(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() >= 4 && "FarfieldBC requires at least 4 variables [p, u, v, w]");

        if (q.size() >= 5) {
            kernels::farfield_kernel<5>(q, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::farfield_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
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
            kernels::farfield_grad_kernel<5>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
        } else {
            kernels::farfield_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
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

    [[nodiscard]] BCType kind() const noexcept override { return BCType::Farfield; }

private:
    FarfieldParams m_p;
};

} // namespace cfd::bc::physical
#pragma once

#include <array>

namespace cfd::solver::bc {

/** 
 * @brief Computes ghost gradient for Dirichlet fixed value BC.
 * Normal component is derived from (u_fixed - u_inner) / rcfn.
 * Tangential component is extrapolated identically from inner cell.
 * @param rcfn_inv 1.0 / dot(x_face - x_cell, n)
 */
[[nodiscard]] constexpr std::array<double, 3> grad_fixed_value_bc(
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double fixed_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const double ddn = (fixed_value - inner_value) * rcfn_inv;
    const double grad_dot_n = inner_gx * nx + inner_gy * ny + inner_gz * nz;
    const double factor = 2.0 * (ddn - grad_dot_n);

    return {
        inner_gx + factor * nx,
        inner_gy + factor * ny,
        inner_gz + factor * nz
    };
}

inline void apply_grad_fixed_value_bc(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double fixed_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const auto g = grad_fixed_value_bc(inner_gx, inner_gy, inner_gz, 
                                       inner_value, fixed_value, 
                                       nx, ny, nz, rcfn_inv);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}

/** 
 * @brief Extrapolates gradient directly from inner cell (zero-gradient or supersonic outflow).
 */
[[nodiscard]] constexpr std::array<double, 3> grad_extrapolation0_bc(
    const double inner_gx, const double inner_gy, const double inner_gz) noexcept {
    return {inner_gx, inner_gy, inner_gz};
}

inline void apply_grad_extrapolation0_bc(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz) noexcept {
    ghost_gx = inner_gx;
    ghost_gy = inner_gy;
    ghost_gz = inner_gz;
}

/** 
 * @brief Computes ghost gradient for Neumann BC with given normal gradient d(u)/dn.
 */
[[nodiscard]] constexpr std::array<double, 3> grad_fixed_gradient_bc(
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double fixed_gradient,
    const double nx, const double ny, const double nz) noexcept {
    
    const double grad_dot_n = inner_gx * nx + inner_gy * ny + inner_gz * nz;
    const double factor = 2.0 * (fixed_gradient - grad_dot_n);

    return {
        inner_gx + factor * nx,
        inner_gy + factor * ny,
        inner_gz + factor * nz
    };
}

inline void apply_grad_fixed_gradient_bc(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double fixed_gradient,
    const double nx, const double ny, const double nz) noexcept {
    
    const auto g = grad_fixed_gradient_bc(inner_gx, inner_gy, inner_gz, 
                                          fixed_gradient, nx, ny, nz);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}

/**
 * @brief Computes ghost gradient for Slip Wall / Symmetry condition.
 * Normal derivative at boundary is estimated from ghost and inner values.
 */
[[nodiscard]] constexpr std::array<double, 3> grad_slip_component_bc(
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double ghost_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const double grad_dot_n = inner_gx * nx + inner_gy * ny + inner_gz * nz;
    const double ddn = 0.5 * (ghost_value - inner_value) * rcfn_inv;
    const double factor = 2.0 * (ddn - grad_dot_n);

    return {
        inner_gx + factor * nx,
        inner_gy + factor * ny,
        inner_gz + factor * nz
    };
}

inline void apply_grad_slip_component_bc(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double ghost_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const auto g = grad_slip_component_bc(inner_gx, inner_gy, inner_gz, 
                                          inner_value, ghost_value, 
                                          nx, ny, nz, rcfn_inv);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}

} // namespace cfd::solver::bc
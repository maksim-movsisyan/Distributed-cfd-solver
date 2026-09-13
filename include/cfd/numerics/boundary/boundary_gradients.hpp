#pragma once

#include <array>

namespace cfd::numerics::boundary {

/** 
 * @brief Computes ghost cell gradient for Dirichlet condition (fixed value).
 * 
 * Normal component is derived from (u_fixed - u_inner) / rcfn,
 * while tangential components are preserved from the interior cell.
 * 
 * @param rcfn_inv Reciprocal of projection distance: 1.0 / dot(x_face - x_cell, n).
 */
[[nodiscard]] constexpr std::array<double, 3> dirichlet_gradient(
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

[[nodiscard]] constexpr std::array<double, 3> dirichlet_gradient(
    const std::array<double, 3>& inner_grad,
    const double inner_value, const double fixed_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    return dirichlet_gradient(inner_grad[0], inner_grad[1], inner_grad[2],
                              inner_value, fixed_value, nx, ny, nz, rcfn_inv);
}

constexpr void apply_dirichlet_gradient(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double fixed_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const auto g = dirichlet_gradient(inner_gx, inner_gy, inner_gz, 
                                      inner_value, fixed_value, 
                                      nx, ny, nz, rcfn_inv);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}


/** 
 * @brief Zero-gradient extrapolation directly from the interior cell (du/dn = 0).
 */
[[nodiscard]] constexpr std::array<double, 3> extrapolate_gradient(
    const double inner_gx, const double inner_gy, const double inner_gz) noexcept {
    return {inner_gx, inner_gy, inner_gz};
}

[[nodiscard]] constexpr std::array<double, 3> extrapolate_gradient(
    const std::array<double, 3>& inner_grad) noexcept {
    return inner_grad;
}

constexpr void apply_extrapolate_gradient(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz) noexcept {
    ghost_gx = inner_gx;
    ghost_gy = inner_gy;
    ghost_gz = inner_gz;
}


/** 
 * @brief Computes ghost cell gradient for Neumann condition with prescribed normal derivative.
 * 
 * @param fixed_gradient Prescribed normal derivative du/dn.
 */
[[nodiscard]] constexpr std::array<double, 3> neumann_gradient(
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

[[nodiscard]] constexpr std::array<double, 3> neumann_gradient(
    const std::array<double, 3>& inner_grad,
    const double fixed_gradient,
    const double nx, const double ny, const double nz) noexcept {
    return neumann_gradient(inner_grad[0], inner_grad[1], inner_grad[2],
                            fixed_gradient, nx, ny, nz);
}

constexpr void apply_neumann_gradient(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double fixed_gradient,
    const double nx, const double ny, const double nz) noexcept {
    
    const auto g = neumann_gradient(inner_gx, inner_gy, inner_gz, 
                                    fixed_gradient, nx, ny, nz);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}


/**
 * @brief Computes ghost gradient for Slip Wall / Symmetry condition.
 * Normal derivative at boundary is estimated from ghost and inner values.
 * 
 * @param rcfn_inv Reciprocal of projection distance: 1.0 / dot(x_face - x_cell, n).
 */
[[nodiscard]] constexpr std::array<double, 3> slip_gradient(
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

[[nodiscard]] constexpr std::array<double, 3> slip_gradient(
    const std::array<double, 3>& inner_grad,
    const double inner_value, const double ghost_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    return slip_gradient(inner_grad[0], inner_grad[1], inner_grad[2],
                         inner_value, ghost_value, nx, ny, nz, rcfn_inv);
}

constexpr void apply_slip_gradient(
    double& ghost_gx, double& ghost_gy, double& ghost_gz,
    const double inner_gx, const double inner_gy, const double inner_gz,
    const double inner_value, const double ghost_value,
    const double nx, const double ny, const double nz, const double rcfn_inv) noexcept {
    
    const auto g = slip_gradient(inner_gx, inner_gy, inner_gz, 
                                 inner_value, ghost_value, 
                                 nx, ny, nz, rcfn_inv);
    ghost_gx = g[0];
    ghost_gy = g[1];
    ghost_gz = g[2];
}

} // namespace cfd::numerics::boundary
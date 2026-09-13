#pragma once

#include <array>

namespace cfd::numerics::boundary {

/**
 * @brief Fill ghost value for Dirichlet fixed value BC (u_face = fixed_value).
 * Linear extrapolation: (u_ghost + u_inner) / 2 = u_fixed
 */
[[nodiscard]] constexpr double dirichlet_value(const double inner_value, 
                                               const double fixed_value) noexcept {
    return 2.0 * fixed_value - inner_value;
}

constexpr void apply_dirichlet_value(double& ghost_value, 
                                     const double inner_value, 
                                     const double fixed_value) noexcept {
    ghost_value = dirichlet_value(inner_value, fixed_value);
}


/**
 * @brief Fill ghost value for zero-gradient Neumann BC (du/dn = 0).
 */
[[nodiscard]] constexpr double extrapolate_value(const double inner_value) noexcept {
    return inner_value;
}

constexpr void apply_extrapolate_value(double& ghost_value, 
                                       const double inner_value) noexcept {
    ghost_value = extrapolate_value(inner_value);
}


/**
 * @brief Fill ghost value for fixed gradient Neumann BC (du/dn = fixed_gradient).
 * @param rcfn Projection of vector (x_face - x_cell) onto outward unit normal n.
 */
[[nodiscard]] constexpr double neumann_value(const double inner_value,
                                             const double fixed_gradient, 
                                             const double rcfn) noexcept {
    return inner_value + 2.0 * fixed_gradient * rcfn;
}

constexpr void apply_neumann_value(double& ghost_value, 
                                   const double inner_value,
                                   const double fixed_gradient, 
                                   const double rcfn) noexcept {
    ghost_value = neumann_value(inner_value, fixed_gradient, rcfn);
}


/**
 * @brief Mirror velocity vector across face normal for Slip Wall / Symmetry (u_n = 0).
 * Assumes (nx, ny, nz) is a normalized unit vector (|n| = 1).
 */
constexpr void apply_slip_component(double& ghost_vx, double& ghost_vy, double& ghost_vz,
                                    const double inner_vx, const double inner_vy, const double inner_vz,
                                    const double nx, const double ny, const double nz) noexcept {
    const double vn2 = 2.0 * (inner_vx * nx + inner_vy * ny + inner_vz * nz);
    ghost_vx = inner_vx - vn2 * nx;
    ghost_vy = inner_vy - vn2 * ny;
    ghost_vz = inner_vz - vn2 * nz;
}

[[nodiscard]] constexpr std::array<double, 3> slip_component(
    const double vx, const double vy, const double vz,
    const double nx, const double ny, const double nz) noexcept {
    const double vn2 = 2.0 * (vx * nx + vy * ny + vz * nz);
    return {vx - vn2 * nx, vy - vn2 * ny, vz - vn2 * nz};
}

[[nodiscard]] constexpr std::array<double, 3> slip_component(
    const std::array<double, 3>& inner_v,
    const double nx, const double ny, const double nz) noexcept {
    return slip_component(inner_v[0], inner_v[1], inner_v[2], nx, ny, nz);
}

} // namespace cfd::numerics::boundary
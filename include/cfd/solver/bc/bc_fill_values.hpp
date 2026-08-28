#pragma once

#include <array>

namespace cfd::solver::bc {

/**
 * @brief Fill ghost value for Dirichlet fixed value BC (u_face = fixed_value).
 * Linear extrapolation: (u_ghost + u_inner) / 2 = u_fixed
 */
[[nodiscard]] constexpr double fixed_value_bc(const double inner_value, 
                                             const double fixed_value) noexcept {
    return 2.0 * fixed_value - inner_value;
}

inline void apply_fixed_value_bc(double& ghost_value, 
                                 const double inner_value, 
                                 const double fixed_value) noexcept {
    ghost_value = fixed_value_bc(inner_value, fixed_value);
}

/**
 * @brief Fill ghost value for zero-gradient Neumann BC (du/dn = 0).
 */
[[nodiscard]] constexpr double extrapolation0_bc(const double inner_value) noexcept {
    return inner_value;
}

inline void apply_extrapolation0_bc(double& ghost_value, 
                                    const double inner_value) noexcept {
    ghost_value = extrapolation0_bc(inner_value);
}

/**
 * @brief Fill ghost value for fixed gradient Neumann BC (du/dn = fixed_gradient).
 * @param rcfn Projection of vector (x_face - x_cell) onto outward unit normal n.
 */
[[nodiscard]] constexpr double fixed_gradient_bc(const double inner_value,
                                                const double fixed_gradient, 
                                                const double rcfn) noexcept {
    return inner_value + 2.0 * fixed_gradient * rcfn;
}

inline void apply_fixed_gradient_bc(double& ghost_value, 
                                    const double inner_value,
                                    const double fixed_gradient, 
                                    const double rcfn) noexcept {
    ghost_value = fixed_gradient_bc(inner_value, fixed_gradient, rcfn);
}

/**
 * @brief Mirror velocity vector across face normal for Slip Wall / Symmetry (u_n = 0).
 * Assumes (nx, ny, nz) is a normalized unit vector (|n| = 1).
 */
inline void apply_slip_component_bc(double& ghost_vx, double& ghost_vy, double& ghost_vz,
                                    const double inner_vx, const double inner_vy, const double inner_vz,
                                    const double nx, const double ny, const double nz) noexcept {
    const double vn2 = 2.0 * (inner_vx * nx + inner_vy * ny + inner_vz * nz);
    ghost_vx = inner_vx - vn2 * nx;
    ghost_vy = inner_vy - vn2 * ny;
    ghost_vz = inner_vz - vn2 * nz;
}

[[nodiscard]] inline std::array<double, 3> slip_velocity_bc(const std::array<double, 3>& inner_v,
                                                            const double nx, const double ny, const double nz) noexcept {
    const double vn2 = 2.0 * (inner_v[0] * nx + inner_v[1] * ny + inner_v[2] * nz);
    return {inner_v[0] - vn2 * nx, inner_v[1] - vn2 * ny, inner_v[2] - vn2 * nz};
}

} // namespace cfd::solver::bc
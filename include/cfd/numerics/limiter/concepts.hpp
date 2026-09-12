#pragma once

#include <concepts>

namespace cfd::numerics::limiter {

/**
 * @concept MultidimSlopeLimiter
 * @brief Unstructured 2D/3D limiters concept.
 */
template <typename L>
concept MultidimSlopeLimiter = requires(double delta_nb, double delta_face, double eps2) {
    { L::name() } -> std::convertible_to<const char*>;
    { L::phi(delta_nb, delta_face, eps2) } noexcept -> std::same_as<double>;
};

/**
 * @concept ScalarLimiter1D
 * @brief Quasi 1D limiters concept.
 */
template <typename L>
concept ScalarLimiter1D = requires(double r) {
    { L::name() } -> std::convertible_to<const char*>;
    { L::phi(r) } noexcept -> std::same_as<double>;
};

} // namespace cfd::numerics::limiter
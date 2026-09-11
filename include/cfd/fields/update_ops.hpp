#pragma once

#include <cstddef>
#include <span>

#include "cfd/core/types.hpp"

namespace cfd::fields {

/**
 * @brief dst[v][c] = x[v][c] for every variable v, owned cells c.
 */
inline void block_copy(std::span<double* const> dst,
                       std::span<double* const> x,
                       const std::size_t n_owned) noexcept {
    for (std::size_t v = 0; v < dst.size(); ++v) {
        double* CFD_RESTRICT d = dst[v];
        const double* CFD_RESTRICT s = x[v];
        for (std::size_t c = 0; c < n_owned; ++c) {
            d[c] = s[c];
        }
    }
}

/**
 * @brief dst[v][c] = x[v][c] - alpha[c] * r[v][c]   (explicit Euler update).
 */
inline void block_sub_axpy(std::span<double* const> dst,
                           std::span<double* const> x,
                           std::span<double* const> r,
                           const double* CFD_RESTRICT alpha,
                           const std::size_t n_owned) noexcept {
    for (std::size_t v = 0; v < dst.size(); ++v) {
        double* CFD_RESTRICT d = dst[v];
        const double* CFD_RESTRICT xv = x[v];
        const double* CFD_RESTRICT rv = r[v];
        for (std::size_t c = 0; c < n_owned; ++c) {
            d[c] = xv[c] - alpha[c] * rv[c];
        }
    }
}

/**
 * @brief dst[v][c] = c1 * x[v][c] + c2 * (y[v][c] - alpha[c] * r[v][c])   (SSP combine).
 */
inline void block_ssp_combine(std::span<double* const> dst,
                              const double c1,
                              std::span<double* const> x,
                              const double c2,
                              std::span<double* const> y,
                              std::span<double* const> r,
                              const double* CFD_RESTRICT alpha,
                              const std::size_t n_owned) noexcept {
    for (std::size_t v = 0; v < dst.size(); ++v) {
        double* CFD_RESTRICT d = dst[v];
        const double* CFD_RESTRICT xv = x[v];
        const double* CFD_RESTRICT yv = y[v];
        const double* CFD_RESTRICT rv = r[v];
        for (std::size_t c = 0; c < n_owned; ++c) {
            d[c] = c1 * xv[c] + c2 * (yv[c] - alpha[c] * rv[c]);
        }
    }
}

} // namespace cfd::fields

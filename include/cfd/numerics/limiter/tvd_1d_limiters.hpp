#pragma once

#include <algorithm>
#include "cfd/numerics/limiter/concepts.hpp"

namespace cfd::numerics::limiter::tvd1d {

struct Minmod {
    static constexpr const char* name() noexcept { return "MINMOD"; }
    [[nodiscard]] static constexpr double phi(const double r) noexcept {
        return std::max(0.0, std::min(r, 1.0));
    }
};

struct VanLeer {
    static constexpr const char* name() noexcept { return "VAN_LEER"; }
    [[nodiscard]] static constexpr double phi(const double r) noexcept {
        return (r <= 0.0) ? 0.0 : (2.0 * r) / (r + 1.0);
    }
};

struct Superbee {
    static constexpr const char* name() noexcept { return "SUPERBEE"; }
    [[nodiscard]] static constexpr double phi(const double r) noexcept {
        return std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));
    }
};

struct VanAlbada {
    static constexpr const char* name() noexcept { return "VAN_ALBADA"; }
    [[nodiscard]] static constexpr double phi(const double r) noexcept {
        return (r <= 0.0) ? 0.0 : (r * r + r) / (r * r + 1.0);
    }
};

static_assert(ScalarLimiter1D<Minmod>);
static_assert(ScalarLimiter1D<VanLeer>);
static_assert(ScalarLimiter1D<Superbee>);
static_assert(ScalarLimiter1D<VanAlbada>);

} // namespace cfd::solver::limiter::tvd1d
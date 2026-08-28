#pragma once

#include <type_traits>

#if defined(_MSC_VER)
    #define CFD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define CFD_RESTRICT __restrict__
#else
    #define CFD_RESTRICT
#endif

namespace cfd::solver::fields {

/**
 * @brief Primitive variables [p, u, v, w, T] SoA View.
 */
template <typename T>
struct PrimitiveView {
    T* CFD_RESTRICT prs{nullptr}; ///< Pressure [Pa]
    T* CFD_RESTRICT vx{nullptr};  ///< Velocity-x [m/s]
    T* CFD_RESTRICT vy{nullptr};  ///< Velocity-y [m/s]
    T* CFD_RESTRICT vz{nullptr};  ///< Velocity-z [m/s]
    T* CFD_RESTRICT tmp{nullptr}; ///< Temperature [K]

    // Default constructors
    constexpr PrimitiveView() noexcept = default;
    constexpr PrimitiveView(T* p, T* u, T* v, T* w, T* t) noexcept
        : prs(p), vx(u), vy(v), vz(w), tmp(t) {}

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr PrimitiveView(const PrimitiveView<U>& o) noexcept
        : prs(o.prs), vx(o.vx), vy(o.vy), vz(o.vz), tmp(o.tmp) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return PrimitiveView<const std::remove_const_t<T>>{prs, vx, vy, vz, tmp};
    }
};

/**
 * @brief Gradients of primitive variables SoA View with planar stride.
 */
template <typename T>
struct PrimitiveGradView {
    std::size_t stride{0};
    T* CFD_RESTRICT prs_grad{nullptr};
    T* CFD_RESTRICT vx_grad{nullptr};
    T* CFD_RESTRICT vy_grad{nullptr};
    T* CFD_RESTRICT vz_grad{nullptr};
    T* CFD_RESTRICT tmp_grad{nullptr};

    constexpr PrimitiveGradView() noexcept = default;
    constexpr PrimitiveGradView(std::size_t s, T* p, T* u, T* v, T* w, T* t) noexcept
        : stride(s), prs_grad(p), vx_grad(u), vy_grad(v), vz_grad(w), tmp_grad(t) {}

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr PrimitiveGradView(const PrimitiveGradView<U>& o) noexcept
        : stride(o.stride), prs_grad(o.prs_grad), vx_grad(o.vx_grad),
          vy_grad(o.vy_grad), vz_grad(o.vz_grad), tmp_grad(o.tmp_grad) {}

    [[nodiscard]] inline T& dprs_dx(const std::size_t idx) const noexcept { return prs_grad[idx]; }
    [[nodiscard]] inline T& dprs_dy(const std::size_t idx) const noexcept { return prs_grad[stride + idx]; }
    [[nodiscard]] inline T& dprs_dz(const std::size_t idx) const noexcept { return prs_grad[2 * stride + idx]; }

    [[nodiscard]] inline T& dvx_dx(const std::size_t idx) const noexcept { return vx_grad[idx]; }
    [[nodiscard]] inline T& dvx_dy(const std::size_t idx) const noexcept { return vx_grad[stride + idx]; }
    [[nodiscard]] inline T& dvx_dz(const std::size_t idx) const noexcept { return vx_grad[2 * stride + idx]; }

    [[nodiscard]] inline T& dvy_dx(const std::size_t idx) const noexcept { return vy_grad[idx]; }
    [[nodiscard]] inline T& dvy_dy(const std::size_t idx) const noexcept { return vy_grad[stride + idx]; }
    [[nodiscard]] inline T& dvy_dz(const std::size_t idx) const noexcept { return vy_grad[2 * stride + idx]; }

    [[nodiscard]] inline T& dvz_dx(const std::size_t idx) const noexcept { return vz_grad[idx]; }
    [[nodiscard]] inline T& dvz_dy(const std::size_t idx) const noexcept { return vz_grad[stride + idx]; }
    [[nodiscard]] inline T& dvz_dz(const std::size_t idx) const noexcept { return vz_grad[2 * stride + idx]; }

    [[nodiscard]] inline T& dtmp_dx(const std::size_t idx) const noexcept { return tmp_grad[idx]; }
    [[nodiscard]] inline T& dtmp_dy(const std::size_t idx) const noexcept { return tmp_grad[stride + idx]; }
    [[nodiscard]] inline T& dtmp_dz(const std::size_t idx) const noexcept { return tmp_grad[2 * stride + idx]; }

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return PrimitiveGradView<const std::remove_const_t<T>>{
            stride, prs_grad, vx_grad, vy_grad, vz_grad, tmp_grad
        };
    }
};

/**
 * @brief Conservative state variables [rho, rhou, rhov, rhow, E] SoA View.
 */
template <typename T>
struct ConservativeView {
    T* CFD_RESTRICT rho{nullptr};
    T* CFD_RESTRICT rhou{nullptr};
    T* CFD_RESTRICT rhov{nullptr};
    T* CFD_RESTRICT rhow{nullptr};
    T* CFD_RESTRICT rhoE{nullptr};

    constexpr ConservativeView() noexcept = default;
    constexpr ConservativeView(T* r, T* ru, T* rv, T* rw, T* re) noexcept
        : rho(r), rhou(ru), rhov(rv), rhow(rw), rhoE(re) {}

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr ConservativeView(const ConservativeView<U>& o) noexcept
        : rho(o.rho), rhou(o.rhou), rhov(o.rhov), rhow(o.rhow), rhoE(o.rhoE) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return ConservativeView<const std::remove_const_t<T>>{rho, rhou, rhov, rhow, rhoE};
    }
};

/**
 * @brief Residual / RHS vectors SoA View.
 */
template <typename T>
struct ResidualView {
    T* CFD_RESTRICT res1{nullptr};
    T* CFD_RESTRICT res2{nullptr};
    T* CFD_RESTRICT res3{nullptr};
    T* CFD_RESTRICT res4{nullptr};
    T* CFD_RESTRICT res5{nullptr};

    constexpr ResidualView() noexcept = default;
    constexpr ResidualView(T* r1, T* r2, T* r3, T* r4, T* r5) noexcept
        : res1(r1), res2(r2), res3(r3), res4(r4), res5(r5) {}

    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr ResidualView(const ResidualView<U>& o) noexcept
        : res1(o.res1), res2(o.res2), res3(o.res3), res4(o.res4), res5(o.res5) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return ResidualView<const std::remove_const_t<T>>{res1, res2, res3, res4, res5};
    }
};

// Typedef aliases
using ConstPrimitiveView     = PrimitiveView<const double>;
using ConstConservativeView  = ConservativeView<const double>;
using ConstResidualView      = ResidualView<const double>;
using ConstPrimitiveGradView = PrimitiveGradView<const double>;

} // namespace cfd::solver::fields
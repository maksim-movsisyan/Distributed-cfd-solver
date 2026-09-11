#pragma once

#include <type_traits>
#include "cfd/core/types.hpp"

namespace cfd::fields {

/**
 * @brief Scalar variable view.
 */
template <typename T>
struct ScalarView {
    T* CFD_RESTRICT data{nullptr};

    constexpr ScalarView() noexcept = default;
    constexpr explicit ScalarView(T* d) noexcept : data(d) {}

    [[nodiscard]] inline T& operator[](std::size_t i) const noexcept { return data[i]; }

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr ScalarView(const ScalarView<U>& o) noexcept
        : data(o.data) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return ScalarView<const std::remove_const_t<T>>{data};
    }
};


/**
 * @brief Vector variable view, SoA.
 */
template <typename T>
struct VectorView {
    T* CFD_RESTRICT data_x{nullptr};
    T* CFD_RESTRICT data_y{nullptr};
    T* CFD_RESTRICT data_z{nullptr};

    constexpr VectorView() noexcept = default;
    constexpr VectorView(T* px, T* py, T* pz) noexcept : data_x(px), data_y(py), data_z(pz) {}

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr VectorView(const VectorView<U>& o) noexcept
        : data_x(o.data_x), data_y(o.data_y), data_z(o.data_z) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return VectorView<const std::remove_const_t<T>>{data_x, data_y, data_z};
    }
};


/**
 * @brief Tensor variable view, SoA.
 */
template <typename T>
struct TensorView {
    T* CFD_RESTRICT data_xx{nullptr}; T* CFD_RESTRICT data_xy{nullptr}; T* CFD_RESTRICT data_xz{nullptr};
    T* CFD_RESTRICT data_yx{nullptr}; T* CFD_RESTRICT data_yy{nullptr}; T* CFD_RESTRICT data_yz{nullptr};
    T* CFD_RESTRICT data_zx{nullptr}; T* CFD_RESTRICT data_zy{nullptr}; T* CFD_RESTRICT data_zz{nullptr};

    constexpr TensorView() noexcept = default;
    
    constexpr TensorView(
        T* pxx, T* pxy, T* pxz,
        T* pyx, T* pyy, T* pyz,
        T* pzx, T* pzy, T* pzz
    ) noexcept : 
        data_xx(pxx), data_xy(pxy), data_xz(pxz),
        data_yx(pyx), data_yy(pyy), data_yz(pyz),
        data_zx(pzx), data_zy(pzy), data_zz(pzz) {}

    // Implicit conversion from Non-Const View to Const View
    template <typename U>
        requires (std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> && std::is_const_v<T>)
    constexpr TensorView(const TensorView<U>& o) noexcept
        : data_xx(o.data_xx), data_xy(o.data_xy), data_xz(o.data_xz),
          data_yx(o.data_yx), data_yy(o.data_yy), data_yz(o.data_yz),
          data_zx(o.data_zx), data_zy(o.data_zy), data_zz(o.data_zz) {}

    [[nodiscard]] constexpr auto as_const() const noexcept {
        return TensorView<const std::remove_const_t<T>>{
            data_xx, data_xy, data_xz,
            data_yx, data_yy, data_yz,
            data_zx, data_zy, data_zz
        };
    }
};






// ---------------- PHYSICS SPECIFIC FIELD VIEWS MUST BE MOVED FROM GENERAL MODULE ----------------

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

} // namespace cfd::fields
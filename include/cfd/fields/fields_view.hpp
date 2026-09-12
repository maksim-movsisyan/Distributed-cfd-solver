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

} // namespace cfd::fields
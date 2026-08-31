#pragma once

#include <concepts>

namespace cfd::solver::riemann {

template <typename Flux, typename EOS>
concept RiemannSolver = requires(
    const EOS& eos,
    const double* UL, 
    const double* UR,
    double nx, double ny, double nz,
    double area,
    double* F,
    double& smax) {
    { Flux::name() } noexcept -> std::same_as<const char*>;

    { Flux::face_flux(eos, UL, UR, nx, ny, nz, area, F, smax) } noexcept -> std::same_as<void>;
};


} // namespace cfd::solver::riemann
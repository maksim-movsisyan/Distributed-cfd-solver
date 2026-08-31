#pragma once

#include "cfd/core/types.hpp"
#include "cfd/solver/eos/eos_concept.hpp"

namespace cfd::solver::eos {

// ============================================================================
// Conserved State Decompositions U = [rho, rho*u, rho*v, rho*w, E]^T
// ============================================================================

// Extract specific internal energy e = (E - 0.5 * rho * |v|^2) / rho [J / kg]
[[nodiscard]] inline double specific_internal_energy(const double U[constants::kNumVars]) noexcept {
    const double rho = U[0];
    const double inv_rho = 1.0 / rho;
    const double q2 = (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) * (inv_rho * inv_rho);
    return (U[4] * inv_rho) - 0.5 * q2;
}

// Static pressure from conserved state: p = p(rho, e) [Pa]
template <EquationOfState EOS>
[[nodiscard]] inline double pressure(const EOS& eos, const double U[constants::kNumVars]) noexcept {
    const double rho = U[0];
    const double e = specific_internal_energy(U);
    return eos.pressure_rhoe(rho, e);
}

// Static temperature from conserved state: T = T(rho, e) [K]
template <EquationOfState EOS>
[[nodiscard]] inline double temperature(const EOS& eos, const double U[constants::kNumVars]) noexcept {
    const double rho = U[0];
    const double e = specific_internal_energy(U);
    return eos.temperature_rhoe(rho, e);
}

// Speed of sound from conserved state: a = a(rho, p) [m / s]
template <EquationOfState EOS>
[[nodiscard]] inline double sound_speed(const EOS& eos, const double U[constants::kNumVars]) noexcept {
    const double rho = U[0];
    const double p = pressure(eos, U);
    return eos.sound_speed_rhop(rho, p);
}

// Total specific enthalpy H = (E + p) / rho [J / kg]
template <EquationOfState EOS>
[[nodiscard]] inline double total_enthalpy(const EOS& eos, const double U[constants::kNumVars]) noexcept {
    const double p = pressure(eos, U);
    return (U[4] + p) / U[0];
}

// ============================================================================
// Conserved -> Primitive Variables Transformations
// ============================================================================

// Conserved state U -> Primitive tuple (rho, u, v, w, p)
template <EquationOfState EOS>
inline void conserved_to_primitives_rhop(const EOS& eos,
                                         const double U[constants::kNumVars],
                                         double& rho, double& u,
                                         double& v, double& w,
                                         double& p) noexcept {
    rho = U[0];
    const double inv_rho = 1.0 / rho;
    u = U[1] * inv_rho;
    v = U[2] * inv_rho;
    w = U[3] * inv_rho;
    
    const double q2 = u * u + v * v + w * w;
    const double e = (U[4] * inv_rho) - 0.5 * q2;
    p = eos.pressure_rhoe(rho, e);
}

// Conserved state U -> Primitive tuple (p, u, v, w, T)
template <EquationOfState EOS>
inline void conserved_to_primitives_pT(const EOS& eos,
                                       const double U[constants::kNumVars],
                                       double& p, double& u,
                                       double& v, double& w,
                                       double& T) noexcept {
    const double rho = U[0];
    const double inv_rho = 1.0 / rho;
    u = U[1] * inv_rho;
    v = U[2] * inv_rho;
    w = U[3] * inv_rho;

    const double q2 = u * u + v * v + w * w;
    const double e = (U[4] * inv_rho) - 0.5 * q2;
    p = eos.pressure_rhoe(rho, e);
    T = eos.temperature_rhoe(rho, e);
}

// ============================================================================
// Primitive -> Conserved Variables Transformations
// ============================================================================

// 1. Primitive tuple (rho, u, v, w, p) -> Conserved vector U
template <EquationOfState EOS>
inline void primitives_rhop_to_conserved(const EOS& eos,
                                         const double rho, const double u,
                                         const double v, const double w,
                                         const double p,
                                         double U[constants::kNumVars]) noexcept {
    const double T = eos.temperature_rhop(rho, p);
    const double e = eos.internal_energy_Tp(T, p);
    const double ke = 0.5 * rho * (u * u + v * v + w * w);

    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = rho * e + ke;
}

// 2. Aerodynamic primitive tuple Q = (p, u, v, w, T) -> Conserved vector U
template <EquationOfState EOS>
inline void primitives_pT_to_conserved(const EOS& eos,
                                       const double p, const double u,
                                       const double v, const double w,
                                       const double T,
                                       double U[constants::kNumVars]) noexcept {
    const double rho = eos.density_Tp(T, p);
    const double e = eos.internal_energy_Tp(T, p);
    const double ke = 0.5 * rho * (u * u + v * v + w * w);

    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = rho * e + ke;
}

} // namespace cfd::solver::eos
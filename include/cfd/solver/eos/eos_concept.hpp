#pragma once

#include <concepts>

namespace cfd::solver::eos {

// ============================================================================
// C++20 Concept: General 2-Parameter Thermodynamic Equation of State
// ============================================================================
template <typename T>
concept EquationOfState = requires(const T eos, double rho, double p, double T_val, double e, double H, double v2) {
    { eos.name() } -> std::convertible_to<const char*>;
    { eos.gamma() } -> std::convertible_to<double>;
    { eos.gas_constant() } -> std::convertible_to<double>;

    // Heat capacities (2-parameter general form)
    { eos.cv_Tp(T_val, p) } -> std::convertible_to<double>;
    { eos.cp_Tp(T_val, p) } -> std::convertible_to<double>;

    // Density and Temperature mappings
    { eos.density_Tp(T_val, p) } -> std::convertible_to<double>;
    { eos.temperature_rhoe(rho, e) } -> std::convertible_to<double>;
    { eos.temperature_rhop(rho, p) } -> std::convertible_to<double>;

    // Caloric functions (Internal Energy & Enthalpy)
    { eos.internal_energy_Tp(T_val, p) } -> std::convertible_to<double>;
    { eos.enthalpy_Tp(T_val, p) } -> std::convertible_to<double>;

    // Thermal Equation of State (Pressure)
    { eos.pressure_rhoe(rho, e) } -> std::convertible_to<double>;
    { eos.pressure_rhoT(rho, T_val) } -> std::convertible_to<double>;

    // Acoustic wave speeds
    { eos.sound_speed_rhop(rho, p) } -> std::convertible_to<double>;
    { eos.sound_speed_Tp(T_val, p) } -> std::convertible_to<double>;
    { eos.sound_speed_Hv2(H, v2) } -> std::convertible_to<double>;

    // Total energy & enthalpy formulations
    { eos.total_energy_v2Tp(v2, T_val, p) } -> std::convertible_to<double>;
    { eos.total_enthalpy_v2Tp(v2, T_val, p) } -> std::convertible_to<double>;
};

} //namespace cfd::solver::eos
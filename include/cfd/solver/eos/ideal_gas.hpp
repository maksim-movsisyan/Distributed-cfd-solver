#pragma once

#include <cmath>
#include "cfd/solver/eos/eos_concept.hpp"

namespace cfd::solver::eos {

// ============================================================================
// Ideal Gas Policy (Calorically Perfect Gas)
// ============================================================================
struct IdealGas {
    double gamma_val = 1.4;        // Ratio of specific heats [-]
    double R_val = 287.052874;     // Specific gas constant [J / (kg K)]

    [[nodiscard]] static constexpr const char* name() noexcept { return "IDEAL_GAS"; }

    [[nodiscard]] constexpr double gamma() const noexcept { return gamma_val; }
    [[nodiscard]] constexpr double gas_constant() const noexcept { return R_val; }
    [[nodiscard]] constexpr double gm1() const noexcept { return gamma_val - 1.0; }

    // Auxiliary constant heat capacities for calorically perfect gas
    [[nodiscard]] constexpr double cv() const noexcept { return R_val / (gamma_val - 1.0); }
    [[nodiscard]] constexpr double cp() const noexcept { return gamma_val * cv(); }

    // Polymorphic 2-parameter heat capacities
    [[nodiscard]] inline double cv_Tp(const double /*T*/, const double /*p*/) const noexcept { return cv(); }
    [[nodiscard]] inline double cp_Tp(const double /*T*/, const double /*p*/) const noexcept { return cp(); }

    // Density mapping
    [[nodiscard]] inline double density_Tp(const double T, const double p) const noexcept {
        return p / (R_val * T);
    }

    // Caloric state functions
    [[nodiscard]] inline double internal_energy_Tp(const double T, const double /*p*/) const noexcept {
        return cv() * T;
    }

    [[nodiscard]] inline double enthalpy_Tp(const double T, const double /*p*/) const noexcept {
        return cp() * T;
    }

    // Temperature mappings
    [[nodiscard]] inline double temperature_rhoe(const double /*rho*/, const double e) const noexcept {
        return e / cv();
    }

    [[nodiscard]] inline double temperature_rhop(const double rho, const double p) const noexcept {
        return p / (rho * R_val);
    }

    // Pressure mappings
    [[nodiscard]] inline double pressure_rhoe(const double rho, const double e) const noexcept {
        return gm1() * rho * e;
    }

    [[nodiscard]] inline double pressure_rhoT(const double rho, const double T) const noexcept {
        return rho * R_val * T;
    }

    // Sound speed formulations
    [[nodiscard]] inline double sound_speed_rhop(const double rho, const double p) const noexcept {
        return std::sqrt(gamma_val * p / rho);
    }

    [[nodiscard]] inline double sound_speed_Tp(const double T, const double /*p*/) const noexcept {
        return std::sqrt(gamma_val * R_val * T);
    }

    [[nodiscard]] inline double sound_speed_Hv2(const double H, const double v2) const noexcept {
        const double a2 = gm1() * (H - 0.5 * v2);
        return std::sqrt(a2 > 0.0 ? a2 : 0.0);
    }

    // Total quantities
    [[nodiscard]] inline double total_energy_v2Tp(const double v2, const double T, const double p) const noexcept {
        return 0.5 * v2 + internal_energy_Tp(T, p);
    }

    [[nodiscard]] inline double total_enthalpy_v2Tp(const double v2, const double T, const double p) const noexcept {
        return 0.5 * v2 + enthalpy_Tp(T, p);
    }
};

// Static compile-time verification that IdealGas satisfies the concept
static_assert(EquationOfState<IdealGas>);

} // namespace cfd::solver::eos
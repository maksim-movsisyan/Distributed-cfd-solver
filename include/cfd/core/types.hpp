#pragma once

#include <cstdint>

#if defined(_MSC_VER)
    #define CFD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define CFD_RESTRICT __restrict__
#else
    #define CFD_RESTRICT
#endif

namespace cfd {

#if defined(CFD_INDEX_64BIT)
using GlobalIndex = std::int64_t;
#else
using GlobalIndex = std::int32_t;
#endif

// Local entity IDs within an MPI rank / partition (up to 2 billion elements)
using LocalIndex = std::int32_t;

// Canonical element topology indices (0..7)
using ElementLocalNode = std::int8_t;

inline constexpr GlobalIndex kInvalidGlobalIndex = -1;
inline constexpr LocalIndex  kInvalidLocalIndex  = -1;


// ============================================================================
// === Centralized physical & numerical constants =============================
// ============================================================================
namespace constants {

// --- ISA standard atmosphere, sea level (default reference state) -----------
inline constexpr double kIsaPressure    = 101325.0;   // [Pa]
inline constexpr double kIsaTemperature = 288.15;     // [K]
inline constexpr double kIsaDensity     = 1.225;      // [kg / m^3]

// --- Calorically perfect air (default working fluid) ------------------------
inline constexpr double kAirGamma       = 1.4;        // [-]
inline constexpr double kAirGasConstant = 287.052874; // [J / (kg K)]

// --- Numerical guards --------------------------------------------------------
inline constexpr double kSpectralRadiusFloor = 1.0e-14;  // [1 / s] keeps dt finite
inline constexpr double kResidualNormFloor   = 1.0e-300; // guards relative norms vs 0 / 0

} // namespace constants

} // namespace cfd
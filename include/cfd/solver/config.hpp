// Strict TOML configuration for the solver: solver.toml (numerics, time,
// output, flow / EOS, initial state) and the boundary condition file (one entry per
// mesh patch). Parsing is fail-fast with precise messages; unknown keys,
// wrong types, missing parameters and patch-coverage mismatches are all hard
// errors.
#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>
#include <string>

#include "cfd/core/types.hpp"
#include "cfd/numerics/gradient/gradient_manager.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/linalg/config.hpp"

namespace cfd::solver {

// --- Equation of state -------------------------------------------------------

enum class EqOfStateType {
    IdealGas,
    // StiffenedGas,
    // RealGas,
    // Incompressible,
};

struct EqOfStateConfig {
    EqOfStateType type = EqOfStateType::IdealGas;

    // Ideal gas parameters
    double gamma = constants::kAirGamma;
    double gas_constant = constants::kAirGasConstant;

    [[nodiscard]] eos::IdealGas create_ideal_gas() const noexcept {
        return eos::IdealGas{gamma, gas_constant};
    }
};


// --- Mean flow model ---------------------------------------------------------

enum class FlowModel {
    InviscidFlow,
    ViscousFlow
};


// --- Numerics / time / output ------------------------------------------------

enum class FluxType {
    HLLC,
};

enum class TimeScheme {
    ForwardEuler,
    SspRk3,
    BackwardEuler,
};

enum class TimeMode {
    Steady,
    DualTime
};

enum class ReconType {
    FirstOrder,
    Muscl,
    MusclDirectional
};

enum class LimiterType {
    None,
    Venkatakrishnan,
    BarthJespersen,
    VanAlbada,
    Minmod1D,
    VanAlbada1D
};

// --- Turbulence modelling ----------------------------------------------------

struct TurbulenceConfig {
    bool enabled = false;                 // selected via [turbulence] model = "SA"
    double nu_inf_ratio = 3.0;            // freestream nu_tilde / nu_molecular [-]
    int max_distance_sweeps = 500;        // wall-distance sweep budget
    double distance_tolerance = 1.0e-8;   // wall-distance relative tolerance
};

enum class TurbulenceModel {
    SA
};


// --- Total config file --------------------------------------------------------


struct SolverConfig {
    // [flow]
    EqOfStateConfig flow;
    double prandtl = constants::kAirPrandtl;            // molecular Prandtl number
    FlowModel flow_model;                               // InviscidFlow / ViscousFlow

    // [initial] — uniform freestream state
    double init_rho = constants::kIsaDensity;
    double init_p = constants::kIsaPressure;
    std::array<double, 3> init_velocity = {0.0, 0.0, 0.0};

    // [numerics]
    numerics::gradient::GradientType gradient = numerics::gradient::GradientType::GreenGaussFace;
    FluxType flux = FluxType::HLLC;
    ReconType reconstruction = ReconType::FirstOrder;
    LimiterType limiter = LimiterType::Venkatakrishnan;
    double limiter_venkat_k = 0.5;                      // Venkatakrishnan smoothing coefficient

    // [turbulence]
    TurbulenceModel turbulence_model;
    TurbulenceConfig turbulence;

    // [time]
    TimeScheme scheme = TimeScheme::SspRk3;
    double cfl = 0.4;
    std::int64_t max_iterations = 10000;
    double residual_tolerance = 1.0e-10; // relative L2 drop
    TimeMode time_mode = TimeMode::Steady;
    //for unsteady time mode:
    double dt = 1e-6;
    std::int64_t max_time_steps = 1;
    std::int64_t bdf_order = 2;

    

    // [linalg]
    linalg::SolverParams linear_solver_params;

    // [output]
    std::string output_dir = "out/solver";
    std::int64_t field_interval = 200;    // iterations between VTU dumps (0 = off)
    std::int64_t residual_interval = 20;  // iterations between log lines
};

// Parses the solver configuration file. MUST be called collectively; any
// violation aborts all ranks via mpi::fatal.
SolverConfig parse_solver_config(const std::string& path, MPI_Comm comm);
} // namespace cfd::solver
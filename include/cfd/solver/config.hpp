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
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/bc/bc.hpp"

namespace cfd::solver {

// --- Boundary conditions ----------------------------------------------------
// One parsed [[boundary_condition]] table.
struct BCDescriptor {
    bc::BCType type = bc::BCType::Symmetry;
    bc::InflowMode inflow_mode{bc::InflowMode::Velocity};
    int patch_id = -1;

    // pressure and temperature
    double p{constants::kIsaPressure};
    double t{constants::kIsaTemperature};
    double tmp_grad{0.0}; ///< Normal temperature gradient dT/dn [K/m]

    // velocity vector
    std::array<double, 3> velocity{0.0, 0.0, 0.0};

    // mach number, angel of atack, slip angel and direction vector
    double mach{0.0};
    double alpha_deg{0.0};
    double beta_deg{0.0};
    std::array<double, 3> direction{1.0, 0.0, 0.0};
    
};

struct BoundaryConfig {
    // Exactly one entry per mesh patch, indexed strictly by patch_id [0, n_patches)
    std::vector<BCDescriptor> patches;
};

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

    // [output]
    std::string output_dir = "out/solver";
    std::int64_t field_interval = 200;    // iterations between VTU dumps (0 = off)
    std::int64_t residual_interval = 20;  // iterations between log lines
};

// Parses the solver configuration file. MUST be called collectively; any
// violation aborts all ranks via mpi::fatal.
SolverConfig parse_solver_config(const std::string& path, MPI_Comm comm);

// Parses the boundary condition file and cross-validates it against the mesh:
// every patch must carry exactly one condition and no unknown patch ids may
// appear. MUST be called collectively.
BoundaryConfig parse_boundary_config(const std::string& path,
                                     const mesh::MeshPart& mp,
                                     MPI_Comm comm);

} // namespace cfd::solver
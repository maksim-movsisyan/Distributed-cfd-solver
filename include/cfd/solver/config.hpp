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

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/bc/bc.hpp"

namespace cfd::solver {

// --- Boundary conditions ----------------------------------------------------
// One parsed [[boundary_condition]] table.
struct BCDescriptor {
    bc::BCType type = bc::BCType::Symmetry;
    int patch_id = -1;

    // SUPERSONIC_INLET/_MACH: static p [Pa], static T [K], Mach number and a
    // unit-normalized flow direction.
    // NONE - velocity vector given
    // _MACH - mach number and velocity direction given
    double p = 101325.0;
    double t = 300.0;
    double mach = 2.0;
    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> direction = {1.0, 0.0, 0.0};

    // FARFIELD/_MACH: Mach, static p [Pa], static T [K]; alpha/beta [deg] rotate
    // the freestream: v = V [cos(alpha) cos(beta), sin(beta),
    //                        sin(alpha) cos(beta)]  (x streamwise, z up).
    // NONE - velocity vector given
    // _MACH - mach number and velocity direction given
    double mach_inf = 2.0;
    double p_inf = 101325.0;
    double t_inf = 300.0;
    std::array<double, 3> velocity_inf = {0.0, 0.0, 0.0};
    double alpha = 0.0; // Angle of attack
    double beta = 0.0;  // Sideslip angle
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
    double gamma = 1.4;
    double gas_constant = 287.052874;

    [[nodiscard]] eos::IdealGas create_ideal_gas() const noexcept {
        return eos::IdealGas{gamma, gas_constant};
    }
};

// --- Numerics / time / output ------------------------------------------------

enum class FluxType {
    HLLC,
};

enum class TimeScheme {
    ForwardEuler,
    SspRk3,
};

struct SolverConfig {
    // [flow]
    EqOfStateConfig flow;

    // [initial] — uniform freestream state
    double init_rho = 1.225;
    double init_p = 101325.0;
    std::array<double, 3> init_velocity = {0.0, 0.0, 0.0};

    // [numerics]
    FluxType flux = FluxType::HLLC;

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
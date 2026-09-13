#pragma once 

#include <array>
#include <string>
#include "cfd/mesh/localmesh.hpp"

#include <mpi.h>

#include "cfd/core/types.hpp"
#include "cfd/bc/physical/physical_bc.hpp"

namespace cfd::bc { 

// One parsed [[boundary_condition]] table.
struct BCDescriptor {
    physical::BCType type = physical::BCType::Symmetry;
    physical::InflowMode inflow_mode{physical::InflowMode::Velocity};
    int patch_id = -1;

    // pressure and temperature
    double p{constants::kIsaPressure};
    double t{constants::kIsaTemperature};
    double tmp_grad{0.0};   ///< Normal temperature gradient dT/dn [K/m]

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

// Parses the boundary condition file and cross-validates it against the mesh:
// every patch must carry exactly one condition and no unknown patch ids may
// appear. MUST be called collectively.
BoundaryConfig parse_boundary_config(const std::string& path,
                                     const mesh::MeshPart& mp,
                                     MPI_Comm comm);

} // namespace cfd::bc
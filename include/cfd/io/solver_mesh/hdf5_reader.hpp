#pragma once

#include <string>

#include <mpi.h>

#include "cfd/mesh/localmesh.hpp"

namespace cfd::io::solver_mesh {

// Loads a fully processed MeshPart (topology, geometry, comm maps, patches)
// from a parallel topology-aware HDF5 mesh container into memory.
//
// Topology-Aware Invariant:
//  - Automatically reads /partition/rank2part mapping and slices hyperslabs
//    strictly by partition ID, guaranteeing identical memory locality as produced
//    by the preprocessor.
//
// MUST be called collectively by all ranks in MPI_COMM_WORLD.
void import_mesh_hdf5(
    mesh::MeshPart& mp,
    const std::string& filepath,
    MPI_Comm comm);

} // namespace cfd::io::solver_mesh
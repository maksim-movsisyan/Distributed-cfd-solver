#pragma once

#include <string>
#include <mpi.h>
#include "cfd/mesh/localmesh.hpp"

namespace cfd::io::vtk {

// Exports MeshPart volume and boundary mesh into VTK XML format (.vtu + .pvtu).
//
// Generates two datasets in `outdir`:
//  1. `<stem>.pvtu` + `<stem>_XXXXX.vtu`: Owned volume cells with CellData:
//     - rank (MPI Rank)
//     - global_id (Global Cell GID)
//     - local_id (Reordered local index [0, n_own))
//     - volume (Cell volume [m^3])
//  2. `<stem>_bnd.pvtu` + `<stem>_bnd_XXXXX.vtu`: Boundary surface faces with CellData:
//     - rank (MPI Rank)
//     - patch_id (Boundary patch ID)
//     - area (Face surface area [m^2])
//
// MUST be called collectively by all ranks in MPI_COMM_WORLD.
void write_vtu(
    const mesh::MeshPart& mp,
    const std::string& outdir,
    const std::string& stem,
    MPI_Comm comm);

} // namespace cfd::io::vtk
#pragma once
// Reading of the custom HDF5 file — exactly the routine the solver will use
// to load its mesh part (all connectivity, ghost maps, geometry, patches).

#include <hdf5.h>

#include <string>

#include "cfd/io/solver_mesh/writer.hpp"
#include "cfd/mesh/localmesh.hpp"

// Read the global metadata (identical on all ranks).
GlobalMeta load_global_meta(hid_t file);

// Read own part (by communicator rank).
void load_partition(hid_t file, int rank, const GlobalMeta& gm, MeshPart& mp);

// Verify the loaded part: ghost-map handshake with the neighbours
// (order and content), the global volume sum, BC coverage.
// Returns true on success; prints diagnostics.
bool verify_partition(const MeshPart& mp, const GlobalMeta& gm);

// Verbose metadata printing (rank 0 plus a short line per rank).
void print_mesh_stats(const MeshPart& mp, const GlobalMeta& gm);

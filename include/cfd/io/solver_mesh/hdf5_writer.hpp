#pragma once

#include <string>
#include <mpi.h>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/partition/partition.hpp"

namespace cfd::io::solver_mesh {

// Writes the complete MeshPart (topology, geometry, comm maps, patches) to a single shared
// HDF5 file using parallel MPI-IO collective operations (H5FD_MPIO_COLLECTIVE).
//
// Topology-Aware Ordering:
//  - Hyperslab offsets in global datasets are ordered by partition ID (`part_id = pr.rank2part[rank]`),
//    ensuring physically contiguous subdomains remain adjacent in the on-disk storage layout.
//
// MUST be called collectively by all ranks in `MPI_COMM_WORLD`.
void export_mesh_hdf5(
    const mesh::MeshPart& mp,
    const partition::PartitionResult& pr,
    const std::string& filepath,
    MPI_Comm comm);

} // namespace cfd::mesh
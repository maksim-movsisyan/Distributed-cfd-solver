#pragma once

#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

// Runs an exhaustive validation of MeshPart invariants and cross-rank MPI symmetry.
// Logs full diagnostic statistics (load balancing, volume metrics, communication profile).
// Aborts via mpi::fatal if any topological or geometric contract is violated.
void validate_and_log_meshpart(const MeshPart& mp);

} // namespace cfd::mesh
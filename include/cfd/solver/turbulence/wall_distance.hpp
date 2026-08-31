// Iterative wall-distance computation for turbulence models (SA, SST blending).
//
// Solves the Eikonal approximation d = min over neighbours (d_nb + |dx|) by
// alternating forward/reverse Gauss-Seidel sweeps over the vertex adjacency,
// with a one-hop halo exchange of the distance field between sweeps. The
// RCM/SFC cache-locality reordering applied by the partitioner makes the local
// index order monotone along mesh directions, so a handful of alternating
// sweeps propagates distance information across the partition (the classical
// fast-sweeping-method effect). Cells touching a no-slip wall patch are
// initialized with the exact centroid-to-wall-face distance; everything else
// starts at +infinity and relaxes down.
//
// One-time initialization cost only; no per-iteration solver work.
#pragma once

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/halo.hpp"

namespace cfd::solver::turb {

/**
 * @brief Computes the wall distance for owned cells (+ one-hop ghosts).
 *
 * @param mp            Local partition (boundary faces grouped by patch).
 * @param adj           Vertex adjacency (sweep stencil; ghosts included).
 * @param halo          Halo exchanger (one-off distance exchanges).
 * @param comm          MPI communicator (global convergence reduction).
 * @param wall_patch    Per-patch flags: true = no-slip turbulent wall.
 * @param max_sweeps    Sweep-pair budget.
 * @param rel_tol       Convergence: max |delta d| / max(d, eps) below this.
 * @param dist          Output array, size >= n_cells (ghost slots are used as
 *                      exchange scratch and end holding ghost distances).
 */
void solve_wall_distance(const mesh::MeshPart& mp,
                         const gradient::VertexAdjacency& adj,
                         halo::HaloExchanger& halo,
                         const MPI_Comm comm,
                         const std::vector<bool>& wall_patch,
                         const int max_sweeps,
                         const double rel_tol,
                         double* dist);

} // namespace cfd::solver::turb

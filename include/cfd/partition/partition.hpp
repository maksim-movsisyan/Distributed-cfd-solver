#pragma once

// Distributed dual-graph partitioning (dKaMinPar) with hardware topology-aware
// process placement.
//
// Algorithm Overview:
//  1. Runtime Hardware Topology Discovery:
//     - Discovers physical compute nodes via MPI_COMM_TYPE_SHARED;
//     - Classifies ranks by (node_id, local_core_id) to map cluster topology.
//  2. Distributed Graph Partitioning:
//     - Invokes dKaMinPar on the distributed CSR dual graph with k = nprocs;
//     - Produces a globally load-balanced partition minimizing total edge cut.
//  3. Topology-Aware Process Placement:
//     - Maps hierarchical partition blocks to ranks grouped by physical nodes;
//     - Guarantees that the dense partition boundaries remain inside shared-memory /
//       NUMA domains, leaving only minimal edge cuts across physical network cables.
//  4. Metric Evaluation:
//     - Performs a lightweight 1-round halo exchange to compute exact global
//       edge-cut and inter-node network cut.

#include <cstdint>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/faces.hpp"
#include "cfd/mesh/raw_mesh.hpp"

namespace cfd::partition {

using PartitionBlockId = std::uint32_t;

struct DKaMinParOptions {
    PartitionBlockId block_count = 0;   // 0 = auto (m.nprocs)
    double imbalance_tolerance = 0.03;  // 3% standard imbalance for CFD
    int threads_per_rank = 1;           // TBB threads per MPI rank
    int seed = 0;
    bool quiet = true;
};

struct PartitionResult {
    // Local cell target ranks of size n_local_cells (who owns this cell after migration)
    std::vector<int> cell_target_rank;

    // Topology metadata: mapping PartitionBlockId -> Target MPI Rank
    //                  : mapping Local MPI rank -> Target PartitionBlockId
    std::vector<int> part2rank;
    std::vector<PartitionBlockId> rank2part;

    GlobalIndex global_edge_cut = 0; // Total cut across all partitions
    GlobalIndex inter_node_cut = 0;  // Cut crossing physical cluster node boundaries
};

// Partition dual graph cells across ranks with topology-aware locality
PartitionResult partition_cells(
    const mesh::RawMesh& m,
    const mesh::DualGraph& g,
    const DKaMinParOptions& dko
);

} // namespace cfd::partition
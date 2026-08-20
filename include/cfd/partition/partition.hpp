#pragma once
// Dual-graph partitioning (distributed KaMinPar) and assignment of the parts
// to ranks with spatial locality in mind (SFC ordering of the part
// centroids: neighbouring parts map to neighbouring ranks so that halo
// exchanges stay inside a single cluster node/socket).

#include <vector>

#include "cfd/mesh/cgns_reader.hpp"
#include "cfd/mesh/faces.hpp"

struct PartitionResult {
    std::vector<uint32_t> block;  // part id of each local cell (before migration)
    std::vector<int> part2rank;   // part -> final rank
    long long edge_cut = 0;       // global
    double imbalance = 0.0;       // max/avg over cell counts
};

void partition_cells(RawMesh& m, DualGraph& g, PartitionResult& pr, int num_threads);

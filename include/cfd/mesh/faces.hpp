#pragma once
// Distributed construction of face-cell connectivity from cell-node connectivity.
//
// Algorithm (no replicated global arrays anywhere):
//  1. every rank generates the faces of its own cells from the canonical tables;
//  2. records are sent to the owner of the face's minimal node (block
//     distribution of nodes) and deduplicated in a hash table keyed by the sorted
//     node set; a face with two cells is interior, with one it is a boundary face;
//  3. each face is sent to the owner of its smaller cell (block distribution
//     of cells); the dual graph (CSR) for the partitioner is assembled there too.

#include <vector>

#include "cfd/mesh/cgns_reader.hpp"
#include "cfd/mesh/cgnstables.hpp"

// POD face record (for all-to-all).
struct FaceRec {
    FaceKey key;     // sorted global node ids
    int32_t cell_a;  // smaller cell gid
    int32_t cell_b;  // larger cell gid or -1 (boundary face)
};

// Dual graph in CSR form: this rank's cells, adjacency by global gid.
struct DualGraph {
    std::vector<long long> offsets;  // n_local + 1
    std::vector<int32_t> adj;        // neighbour global gids
};

struct FaceStats {
    long long n_faces_g = 0;   // total faces
    long long n_bfaces_g = 0;  // boundary faces
    long long n_ifaces_g = 0;  // interior faces
};

// Global face statistics plus the local face list (faces whose smaller cell
// lives on this rank) and the CSR dual graph.
void build_faces(RawMesh& m, std::vector<FaceRec>& faces, DualGraph& g, FaceStats& st);

#pragma once
// Parallel CGNS reading via PCGNS (cgp_* on top of parallel HDF5).
//
// Supported: 1 base, 1 zone (Unstructured), per-type element sections without
// MIXED: TETRA_4, PYRA_5, PENTA_6, HEXA_8 (volume) + TRI_3, QUAD_4 (boundary,
// for BCs). BAR_* are skipped. BCs: ZoneBC with PointList/PointRange (GridLocation
// = FaceCenter); the name is the FamilyName when present, else the BC node name.

#include <string>
#include <vector>

#include "cfd/mesh/cgnstables.hpp"

struct SectionMeta {
    std::string name;
    CellType type;
    long long start = 0, end = 0;  // global (1-based) element ids of the section
    long long cell_offset = 0;     // volume sections: gid of the section's first cell (0-based)
    int sec_idx = 0;               // original 1-based section index in the CGNS file
};

struct BCMeta {
    std::string name;             // FamilyName or the boco node name
    std::string cgns_type;        // BCType_t string (usually BCUserDefined)
    std::vector<long long> eids;  // global boundary element ids (1-based)
};

// Boundary element (for BC matching): node-set key plus its global id.
struct SurfElem {
    FaceKey key;
    int32_t eid = 0;     // global 1-based element id
    int32_t patch = -1;  // index into patch_list or -1 (not covered by any BC)
};

// Read mesh. Cells are block-distributed by global id (gid is a dense
// renumbering of the volume sections in order); node coordinates are block-distributed.
struct RawMesh {
    int nprocs = 0, rank = 0;
    long long n_cells_g = 0, n_nodes_g = 0;
    std::vector<long long> cell_displ, node_displ;  // size nprocs+1

    // Local cells: cgid strictly increasing within [cell_displ[rank], ...)
    std::vector<int32_t> cgid;    // size n_local
    std::vector<uint8_t> ctype;   // CellType, volume types only
    std::vector<int32_t> cnodes;  // global node ids (0-based), flat over cells

    // Coordinates of this rank's node slice (node_displ[rank]..node_displ[rank+1]),
    // stored as consecutive x,y,z triples.
    std::vector<double> my_node_coords;

    // Parallel slice of the boundary elements (all ranks together read them all).
    std::vector<SurfElem> surf_elems;

    // Replicated metadata.
    std::vector<SectionMeta> vol_secs, surf_secs;
    std::vector<BCMeta> bcs;
    // Global BC patch list: (name, cgns type); the order defines the patch id.
    std::vector<std::pair<std::string, std::string>> patch_list;

    long long n_local() const { return cell_displ[rank + 1] - cell_displ[rank]; }
    long long my_node_begin() const { return node_displ[rank]; }
    long long my_node_end() const { return node_displ[rank + 1]; }
};

// Parallel read (all ranks of the communicator). On file incompatibility it
// prints an error and returns nullptr on every rank.
RawMesh* read_cgns_parallel(const std::string& path);

// Fetch node coordinates by global ids (the node owner answers from its own
// slice). Returns xyz triples in the input-list order (duplicates
// are allowed).
std::vector<double> fetch_coords(RawMesh& m, const std::vector<int32_t>& node_gids);

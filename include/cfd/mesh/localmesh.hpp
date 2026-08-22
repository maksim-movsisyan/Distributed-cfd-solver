#pragma once
// Final rank-local mesh: owned cells plus a one-hop ghost layer (face and
// vertex-sharing neighbours), locality-oriented renumbering, faces and the
// communication maps. This is exactly the structure the solver loads from file.

#include <string>
#include <vector>

#include "cfd/mesh/cgnstables.hpp"

class RawMesh;
struct FaceRec;
struct PartitionResult;

struct MeshPart {
    int rank = 0, nprocs = 0;

    // --- Global metadata (identical on all ranks) ---
    long long n_cells_g = 0, n_faces_g = 0, n_nodes_g = 0, n_bfaces_g = 0;
    double bbox_lo[3] = {0, 0, 0}, bbox_hi[3] = {0, 0, 0};

    // --- Cells: [0, n_own) owned (SFC-ordered), [n_own, n_cells) ghosts ---
    int n_own = 0, n_cells = 0;
    std::vector<uint8_t> cell_type;     // CellType
    std::vector<int32_t> cell_gid;      // global id
    std::vector<int32_t> cell_donor;    // -1 for owned / donor rank for ghosts
    std::vector<int32_t> cell_nodes;    // 8 local node slots (-1 padded)
    std::vector<double> cell_centroid;  // 3 * n_cells
    std::vector<double> cell_volume;    // n_cells

    // --- Nodes: [0, n_nodes_own) used by owned cells, then the rest ---
    int n_nodes_own = 0, n_nodes = 0;
    std::vector<double> node_xyz;   // 3 * n_nodes
    std::vector<int32_t> node_gid;  // global id

    // --- Faces: owner is a local owned cell; sorted by (owner, neigh) ---
    int n_faces = 0;
    std::vector<int32_t> face_owner;    // local index of the owned cell
    std::vector<int32_t> face_neigh;    // local index (owned|ghost) or -1
    std::vector<uint8_t> face_type;     // TRI / QUAD
    std::vector<int32_t> face_nodes;    // 4 local node slots (-1 padded)
    std::vector<double> face_centroid;  // 3 * n_faces
    std::vector<double> face_normal;    // 3 * n_faces, outward from owner
    std::vector<double> face_area;      // n_faces
    std::vector<int32_t> face_patch;    // BC patch id or -1
    std::vector<int32_t> face_donor;    // rank owning the ghost neighbour or -1

    // --- Communication maps (neighbours sorted by rank) ---
    std::vector<int32_t> nb_ranks;
    std::vector<int> recv_offsets;          // n_neighbors+1, into recv_ghost_local
    std::vector<int32_t> recv_ghost_local;  // my ghost indices grouped by neighbour
    std::vector<int> send_offsets;          // n_neighbors+1, into send_owned_local
    std::vector<int32_t> send_owned_local;  // my owned indices needed by neighbours

    // --- BC patches (replicated global list) ---
    struct Patch {
        std::string name, cgns_type;
    };
    std::vector<Patch> patches;
    std::vector<int> patch_face_offsets;  // n_patches+1
    std::vector<int32_t> patch_faces;     // local face indices of the patches

    // --- Preprocessing statistics ---
    long long n_flipped = 0;  // cells with a fixed orientation (global)

    int n_neighbors() const { return static_cast<int>(nb_ranks.size()); }
};

// Structure sanity (invariants); used by tests.
bool meshpart_sane(const MeshPart& mp, std::string& err);

// Full transition from the read mesh to the final rank-local one: migration
// to the final ranks, ghost layer, SFC ordering, renumbering, faces,
// orientation fix. On entry `faces` are the local faces (min cell here);
// on exit they are reordered in sync with mp.face_*.
void build_local_mesh(RawMesh& m, std::vector<FaceRec>& faces, const PartitionResult& pr,
                      MeshPart& mp);

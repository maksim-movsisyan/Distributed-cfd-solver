#pragma once
// Writing of the final custom HDF5 file (a single file per run,
// opened in parallel, no compression).

#include <string>
#include <vector>

#include "cfd/mesh/localmesh.hpp"

struct GlobalMeta {
    int nprocs = 0;
    long long n_cells_g = 0, n_faces_g = 0, n_nodes_g = 0, n_bfaces_g = 0;
    double bbox_lo[3] = {0, 0, 0}, bbox_hi[3] = {0, 0, 0};
    double total_volume = 0.0;  // sum of owned-cell volumes (a control value)
    long long n_flipped = 0;
    long long n_ghost_total = 0;  // sum of ghosts over all ranks
    std::vector<std::string> patch_names, patch_types;
    // Rank ranges in the flat arrays (nprocs+1 entries; filled by the loader).
    std::vector<int64_t> cells_off, nodes_off, faces_off, nb_off, recv_off, send_off,
        patchfaces_off, recvseg_off, sendseg_off;
    std::vector<int64_t> n_own;  // nprocs
};

void write_mesh_file(const std::string& path, const MeshPart& mp, const GlobalMeta& gm);

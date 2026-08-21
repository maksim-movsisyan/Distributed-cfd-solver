#pragma once
// VTU output of the rank-local mesh part to visualize the partitioning in ParaView.
// Each rank writes TWO pieces:
//   <stem>_XXXXX.vtu     — owned volume cells only, CellData {rank, global_id,
//                          local_id} (local_id = rank-local index, SFC-ordered);
//   <stem>_bnd_XXXXX.vtu — boundary faces only, CellData {rank, patch}.
// Ghost cells are not exported (they duplicate neighbouring blocks and were
// verified visually once). Rank 0 writes the .pvtu collections for both.

#include <string>

#include "cfd/mesh/localmesh.hpp"

void write_vtu(const MeshPart& mp, const std::string& outdir, const std::string& stem);

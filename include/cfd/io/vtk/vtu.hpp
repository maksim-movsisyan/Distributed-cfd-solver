#pragma once
// VTU output of the rank-local mesh part to visualize the partitioning in ParaView.
// Each rank writes part_X.vtu (owned cells plus boundary faces carrying
// the patch id; ghost cells are not exported); rank 0 writes the
// part.pvtu collection.

#include <string>

#include "cfd/mesh/localmesh.hpp"

void write_vtu(const MeshPart& mp, const std::string& outdir, const std::string& stem);

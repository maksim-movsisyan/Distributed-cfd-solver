// Cell and face geometry plus a self-check of the canonical face tables.
#pragma once

#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

// Polyhedron volume from cell node coordinates (SoA layout):
// V = (1/3) sum_f (S_f · c_f), where S_f is the outward area normal vector
// and c_f is the face centroid. Exact for convex polyhedra with planar faces.
// x, y, z must contain at least kNodesPerType[t] coordinates.
[[nodiscard]] double poly_cell_volume(CellType t, 
                                      const double* x, 
                                      const double* y, 
                                      const double* z) noexcept;

// Self-check of canonical tables on reference elements (TET/PYRA/PRISM/HEXA):
// verifies positive volume, outward normals for all faces, and orientation inversion.
bool validate_face_tables();

void compute_mesh_geometry(MeshPart& mp);

} //namespace cfd::mesh
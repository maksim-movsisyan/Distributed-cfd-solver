#pragma once
// Cell and face geometry plus a self-check of the canonical face tables.

#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mesh/localmesh.hpp"

// Polyhedron volume from the cell nodes: V = (1/3) sum_f (S_f . c_f),
// where S_f is the face area vector (fan over the canonical node order)
// and c_f the face centroid. Exact for convex cells with planar faces.
// pts holds the cell node coordinates (3 * kNodesPerType values).
double poly_cell_volume(CellType t, const double* pts);

// Final local-mesh geometry: cell centroids/volumes (owned + ghosts),
// face centroids/areas/normals (normal outward from the owner). Verifies
// sign consistency; prints diagnostics and returns false on failure.
bool compute_geometry(MeshPart& mp);

// Self-check of the tables on reference elements (TET/PYRA/PRISM/HEXA):
// positive volume, outward normals of all faces, and a working
// orientation-flip permutation. Call once on rank 0.
bool validate_face_tables();

#pragma once
// Matching of boundary faces against CGNS BC elements (distributed, via the
// face-key hash on the minimal-node owner) and generation of the text
// boundary-condition file for the user.

#include "cfd/mesh/localmesh.hpp"

class RawMesh;
struct FaceRec;

// Fills mp.face_patch, mp.patch_face_offsets, mp.patch_faces.
// faces is the array kept in sync with mp.face_* (see build_local_mesh).
// Returns false if boundary faces without a BC or BC elements without a face exist.
bool match_boundaries(RawMesh& m, MeshPart& mp, std::vector<FaceRec>& faces);

// Write the BC file with GLOBAL per-patch face counts. Must be called by
// ALL ranks (allreduce inside); only rank 0 writes. The user fills in the
// type and the parameters; per-patch local face indices live in the HDF5
// file (patches/*) — this file is for problem setup, not for the solver.
void write_bc_config(const MeshPart& mp, const std::string& path);

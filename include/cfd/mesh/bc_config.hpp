#pragma once

#include <string>
#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

// Generates a human-readable boundary condition configuration template.
// MUST be called on Rank 0 (or internally guarded by rank == 0).
// Output file can be directly edited by the user and loaded into the solver.
void generate_bc_template_config(
    const MeshPart& mp,
    const std::string& output_filepath);

} // namespace cfd::mesh
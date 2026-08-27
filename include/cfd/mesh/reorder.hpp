#pragma once

#include <cstdint>
#include <stdexcept>
#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

enum class ReorderMethod : std::uint8_t {
    NONE = 0,
    HILBERT_SFC = 1,
    RCM = 2
};

inline ReorderMethod parse_reorder_method(const char* arg) {
    if (!arg) {
        throw std::invalid_argument("Missing argument value for reorder method");
    }
    
    std::string_view method_str(arg);

    if (method_str == "NONE") return ReorderMethod::NONE;
    if (method_str == "RCM") return ReorderMethod::RCM;
    if (method_str == "SFC" || method_str == "HILBERT_SFC") return ReorderMethod::HILBERT_SFC;

    throw std::invalid_argument("Unknown reorder method: " + std::string(method_str));
}

inline const char* reorder_method_to_string(ReorderMethod method) {
    switch (method) {
        case ReorderMethod::NONE: return "NONE";
        case ReorderMethod::HILBERT_SFC: return "HILBERT_SFC";
        case ReorderMethod::RCM: return "RCM";
        default: return "UNKNOWN";
    }
}

// Reorders owned cells [0, n_own) to optimize CPU cache locality and matrix bandwidth.
// Ghost cells [n_own, n_cells) remain intact to preserve zero-copy MPI communication maps.
// Updates all cell arrays, face connectivity, face sorting, send_owned_local maps, and BC patches.
void reorder_local_mesh(MeshPart& mp, ReorderMethod method);

} // namespace cfd::mesh
#pragma once

#include <algorithm>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::numerics::boundary {

/**
 * @brief Fast boundary face traversal (topology only: face_idx, in, gh).
 * Does not read normals or centroids from memory.
 */
template <typename Kernel>
inline void for_each_boundary_face(const mesh::MeshPart& m,
                                   const LocalIndex fbeg,
                                   const LocalIndex fend,
                                   Kernel&& kernel) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        kernel(face_idx, in, gh);

        ++f_loc;
    }
}

/**
 * @brief Boundary face traversal with unit normal vector.
 */
template <typename Kernel>
inline void for_each_boundary_face_normal(const mesh::MeshPart& m,
                                          const LocalIndex fbeg,
                                          const LocalIndex fend,
                                          Kernel&& kernel) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr = m.face_normal_z.data();

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        kernel(face_idx, in, gh, nx_ptr[face_idx], ny_ptr[face_idx], nz_ptr[face_idx]);

        ++f_loc;
    }
}

/**
 * @brief Full boundary face traversal for gradient operators.
 * Flat arguments: face_idx, in, gh, nx, ny, nz, rcfn_inv.
 */
template <typename Kernel>
inline void for_each_boundary_face_grad(const mesh::MeshPart& m,
                                 const LocalIndex fbeg,
                                 const LocalIndex fend,
                                 Kernel&& kernel) noexcept {
    const auto beg = static_cast<std::size_t>(fbeg);
    const auto end = static_cast<std::size_t>(fend);
    if (beg >= end) return;

    const auto n_cells = static_cast<std::size_t>(m.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);
    std::size_t f_loc = beg - n_inner_faces;

    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const double* CFD_RESTRICT nx_ptr         = m.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr         = m.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr         = m.face_normal_z.data();

    const double* CFD_RESTRICT fcx_ptr = m.face_centroid_x.data();
    const double* CFD_RESTRICT fcy_ptr = m.face_centroid_y.data();
    const double* CFD_RESTRICT fcz_ptr = m.face_centroid_z.data();

    const double* CFD_RESTRICT ccx_ptr = m.cell_centroid_x.data();
    const double* CFD_RESTRICT ccy_ptr = m.cell_centroid_y.data();
    const double* CFD_RESTRICT ccz_ptr = m.cell_centroid_z.data();

    for (std::size_t face_idx = beg; face_idx < end; ++face_idx) {
        const auto in = static_cast<std::size_t>(face_owner[face_idx]);
        const auto gh = n_cells + f_loc;

        const double nx = nx_ptr[face_idx];
        const double ny = ny_ptr[face_idx];
        const double nz = nz_ptr[face_idx];

        const double rcfx = fcx_ptr[face_idx] - ccx_ptr[in];
        const double rcfy = fcy_ptr[face_idx] - ccy_ptr[in];
        const double rcfz = fcz_ptr[face_idx] - ccz_ptr[in];

        const double rcfn = rcfx * nx + rcfy * ny + rcfz * nz;
        const double rcfn_inv = 1.0 / std::max(rcfn, 1.0e-14);

        kernel(face_idx, in, gh, nx, ny, nz, rcfn_inv);

        ++f_loc;
    }
}

} // namespace cfd::numerics::boundary
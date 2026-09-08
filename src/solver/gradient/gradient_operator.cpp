#include "cfd/solver/gradient/gradient_operator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/localmesh.hpp"



namespace cfd::solver::gradient {

namespace {

// Inverts 3x3 symmetric matrix via adjugate
inline bool invert3x3(const double A[9], double invA[9]) noexcept {
    constexpr double eps = 1.0e-14;

    const double det = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                       A[1] * (A[3] * A[8] - A[5] * A[6]) +
                       A[2] * (A[3] * A[7] - A[4] * A[6]);

    if (std::abs(det) <= eps) {
        std::fill_n(invA, 9, 0.0);
        return false;
    }

    const double invDet = 1.0 / det;

    invA[0] = (A[4] * A[8] - A[5] * A[7]) * invDet;
    invA[1] = (A[2] * A[7] - A[1] * A[8]) * invDet;
    invA[2] = (A[1] * A[5] - A[2] * A[4]) * invDet;

    invA[3] = (A[5] * A[6] - A[3] * A[8]) * invDet;
    invA[4] = (A[0] * A[8] - A[2] * A[6]) * invDet;
    invA[5] = (A[2] * A[3] - A[0] * A[5]) * invDet;

    invA[6] = (A[3] * A[7] - A[4] * A[6]) * invDet;
    invA[7] = (A[1] * A[6] - A[0] * A[7]) * invDet;
    invA[8] = (A[0] * A[4] - A[1] * A[3]) * invDet;

    return true;
}

constexpr std::size_t kMaxBatchVariables    = 16;
constexpr std::size_t kMaxCellFaceNeighbors = 64;
constexpr std::size_t kMaxCellNodeNeighbors = 128;

} // anonymous namespace

// =============================================================================
// Green-Gauss Cell-Based (CB)
// =============================================================================

void GreenGaussCellGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    if (!aux_conn.has_cell_faces()) {
        build_cell_faces_conn(mesh, aux_conn.cell_faces_offsets, aux_conn.cell_faces);
        aux_conn.active_mask = aux_conn.active_mask | mesh::AuxConnType::CellFaces;
    }
}

void GreenGaussCellGradient::apply(const double* const CFD_RESTRICT s, 
                                   double* const CFD_RESTRICT g, 
                                   const std::size_t stride, 
                                   const mesh::MeshPart& mesh, 
                                   const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_faces() && "GreenGaussCellGradient requires CellFaces connectivity!");
    
    const std::size_t n_own = static_cast<std::size_t>(mesh.n_own);
    const std::size_t n_total = static_cast<std::size_t>(mesh.n_cells);
    const std::size_t n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);

    const LocalIndex* CFD_RESTRICT cell_faces_ptr = aux_conn.cell_faces_offsets.data();
    const LocalIndex* CFD_RESTRICT cell_faces     = aux_conn.cell_faces.data();
    const LocalIndex* CFD_RESTRICT face_owner     = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh     = mesh.face_neigh.data();

    const double* CFD_RESTRICT face_area     = mesh.face_area.data();
    const double* CFD_RESTRICT face_normal_x = mesh.face_normal_x.data();
    const double* CFD_RESTRICT face_normal_y = mesh.face_normal_y.data();
    const double* CFD_RESTRICT face_normal_z = mesh.face_normal_z.data();
    const double* CFD_RESTRICT cell_volume   = mesh.cell_volume.data();

    // loop over all local cells
    for (std::size_t idx = 0; idx < n_own; ++idx) {
        const double fi_self = s[idx];

        const std::size_t ptr1 = static_cast<std::size_t>(cell_faces_ptr[idx]);
        const std::size_t ptr2 = static_cast<std::size_t>(cell_faces_ptr[idx + 1]);

        double g_x = 0.0, g_y = 0.0, g_z = 0.0;

        // loop over all local cell faces
        for (std::size_t f_ptr = ptr1; f_ptr < ptr2; ++f_ptr) {
            const std::size_t face_idx = static_cast<std::size_t>(cell_faces[f_ptr]);

            const double area = face_area[face_idx];
            const double nx = face_normal_x[face_idx];
            const double ny = face_normal_y[face_idx];
            const double nz = face_normal_z[face_idx];

            const LocalIndex owner = face_owner[face_idx];
            const LocalIndex neigh = face_neigh[face_idx];
            const double sign      = (static_cast<LocalIndex>(idx) == owner) ? 1.0 : -1.0;

            std::size_t idx2;
            if (neigh >= 0) {
                idx2 = (static_cast<LocalIndex>(idx) == owner) 
                        ? static_cast<std::size_t>(neigh) : static_cast<std::size_t>(owner);
            } else {
                idx2 = n_total + face_idx - n_inner_faces;
            }

            const double fi_face = 0.5 * (fi_self + s[idx2]);

            const double flux = sign * area * fi_face;
            g_x += flux * nx;
            g_y += flux * ny;
            g_z += flux * nz;
        } // end loop over all local cell faces

        const double vol_inv = 1.0 / cell_volume[idx];
        g[idx]              = g_x * vol_inv;
        g[idx + stride]     = g_y * vol_inv;
        g[idx + 2 * stride] = g_z * vol_inv;
    } // end loop over all local cells
}

void GreenGaussCellGradient::apply_set(std::span<const double*> s, 
                                       std::span<double*> g, 
                                       const std::size_t stride,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_faces() && "GreenGaussCellGradient requires CellFaces connectivity!");

    const std::size_t n_vars = s.size();
    assert(n_vars == g.size() && "Mismatched span sizes in apply_set!");
    if (n_vars == 0) return;
    assert(n_vars <= kMaxBatchVariables && "Batch size exceeds stack capacity!");

    const double* CFD_RESTRICT s_ptrs[kMaxBatchVariables];
    double*       CFD_RESTRICT g_ptrs[kMaxBatchVariables];
    for (std::size_t v = 0; v < n_vars; ++v) {
        s_ptrs[v] = s[v];
        g_ptrs[v] = g[v];
    }

    const auto n_own         = static_cast<std::size_t>(mesh.n_own);
    const auto n_total       = static_cast<std::size_t>(mesh.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);

    const LocalIndex* CFD_RESTRICT cell_faces_off = aux_conn.cell_faces_offsets.data();
    const LocalIndex* CFD_RESTRICT cell_faces     = aux_conn.cell_faces.data();
    const LocalIndex* CFD_RESTRICT face_owner     = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh     = mesh.face_neigh.data();

    const double* CFD_RESTRICT face_area     = mesh.face_area.data();
    const double* CFD_RESTRICT face_normal_x = mesh.face_normal_x.data();
    const double* CFD_RESTRICT face_normal_y = mesh.face_normal_y.data();
    const double* CFD_RESTRICT face_normal_z = mesh.face_normal_z.data();
    const double* CFD_RESTRICT cell_volume   = mesh.cell_volume.data();


    for (std::size_t idx = 0; idx < n_own; ++idx) {
        double gx[kMaxBatchVariables] = {0.0};
        double gy[kMaxBatchVariables] = {0.0};
        double gz[kMaxBatchVariables] = {0.0};

        const auto ptr1 = static_cast<std::size_t>(cell_faces_off[idx]);
        const auto ptr2 = static_cast<std::size_t>(cell_faces_off[idx + 1]);

        for (std::size_t f_ptr = ptr1; f_ptr < ptr2; ++f_ptr) {
            const auto face_idx = static_cast<std::size_t>(cell_faces[f_ptr]);

            const double area = face_area[face_idx];
            const double nx   = face_normal_x[face_idx];
            const double ny   = face_normal_y[face_idx];
            const double nz   = face_normal_z[face_idx];

            const LocalIndex owner = face_owner[face_idx];
            const LocalIndex neigh = face_neigh[face_idx];
            const double sign      = (static_cast<LocalIndex>(idx) == owner) ? 1.0 : -1.0;

            const double geom_x = sign * area * nx;
            const double geom_y = sign * area * ny;
            const double geom_z = sign * area * nz;

            std::size_t idx2;
            if (neigh >= 0) {
                idx2 = (static_cast<LocalIndex>(idx) == owner) 
                           ? static_cast<std::size_t>(neigh) 
                           : static_cast<std::size_t>(owner);
            } else {
                idx2 = n_total + face_idx - n_inner_faces;
            }

            for (std::size_t v = 0; v < n_vars; ++v) {
                const double fi_face = 0.5 * (s_ptrs[v][idx] + s_ptrs[v][idx2]);
                gx[v] += geom_x * fi_face;
                gy[v] += geom_y * fi_face;
                gz[v] += geom_z * fi_face;
            }
        }

        const double vol_inv = 1.0 / cell_volume[idx];
        for (std::size_t v = 0; v < n_vars; ++v) {
            g_ptrs[v][idx]              = gx[v] * vol_inv;
            g_ptrs[v][idx + stride]     = gy[v] * vol_inv;
            g_ptrs[v][idx + 2 * stride] = gz[v] * vol_inv;
        }
    }
}


// =============================================================================
// Green-Gauss Face-Based (FB)
// =============================================================================

void GreenGaussFaceGradient::setup(const mesh::MeshPart&, mesh::MeshAuxConnectivity&) {}

void GreenGaussFaceGradient::apply(const double* const CFD_RESTRICT s, double* const CFD_RESTRICT g, const std::size_t stride, 
                                   const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity&) const {
    const std::size_t n_own = static_cast<std::size_t>(mesh.n_own);
    const std::size_t n_total = static_cast<std::size_t>(mesh.n_cells);
    const std::size_t n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT face_area     = mesh.face_area.data();
    const double* CFD_RESTRICT face_normal_x = mesh.face_normal_x.data();
    const double* CFD_RESTRICT face_normal_y = mesh.face_normal_y.data();
    const double* CFD_RESTRICT face_normal_z = mesh.face_normal_z.data();
    const double* CFD_RESTRICT cell_volume   = mesh.cell_volume.data();

    std::fill_n(g, 3 * stride, 0.0);
    
    auto process_face = [&](std::size_t face_idx, LocalIndex neighbor_idx, double* CFD_RESTRICT out_val) {
        
        const LocalIndex owner = face_owner[face_idx];

        const double area = face_area[face_idx];
        const double nx   = face_normal_x[face_idx];
        const double ny   = face_normal_y[face_idx];
        const double nz   = face_normal_z[face_idx];

        const double fi_owner = s[static_cast<std::size_t>(owner)];
        const double fi_neigh = s[static_cast<std::size_t>(neighbor_idx)];
        const double fi_face  = 0.5 * (fi_owner + fi_neigh);

        const double val_x = area * fi_face * nx;
        const double val_y = area * fi_face * ny;
        const double val_z = area * fi_face * nz;

        // Scatter to owner cell 
        const std::size_t l_sz = static_cast<std::size_t>(owner);
        g[l_sz]              += val_x;
        g[l_sz + stride]     += val_y;
        g[l_sz + 2 * stride] += val_z;

        out_val[0] = val_x;
        out_val[1] = val_y;
        out_val[2] = val_z;
    };

    // loop over all inner faces
    for (std::size_t face_idx = 0; face_idx < n_inner_faces; ++face_idx) {
        const LocalIndex neighbor = face_neigh[face_idx];
        double flux[3];
        
        process_face(face_idx, neighbor, flux);

        // Scatter to neighbor cell
        const std::size_t r_sz = static_cast<std::size_t>(neighbor);
        g[r_sz]              -= flux[0];
        g[r_sz + stride]     -= flux[1];
        g[r_sz + 2 * stride] -= flux[2];
    } // end loop over all inner faces

    // loop over all boundary faces
    for (std::size_t face_idx = n_inner_faces; face_idx < n_faces; ++face_idx) {
        const LocalIndex neighbor = static_cast<LocalIndex>(n_total + face_idx - n_inner_faces);
        double dummy_flux[3]; // Stack allocation, optimized away by compiler
        
        process_face(face_idx, neighbor, dummy_flux);
    } // end loop over all inner faces


    // normalize by cell volume
    for (std::size_t idx = 0; idx < n_own; ++idx) {
        const double cv_inv = 1.0 / cell_volume[idx];
        g[idx]              *= cv_inv;
        g[idx + stride]     *= cv_inv;
        g[idx + 2 * stride] *= cv_inv;
    }
}

void GreenGaussFaceGradient::apply_set(std::span<const double*> s, 
                                       std::span<double*> g, 
                                       const std::size_t stride,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity&) const {
    const std::size_t n_vars = s.size();
    assert(n_vars == g.size() && "Mismatched span sizes in apply_set!");
    if (n_vars == 0) return;
    assert(n_vars <= kMaxBatchVariables && "Batch size exceeds stack capacity!");

    const double* CFD_RESTRICT s_ptrs[kMaxBatchVariables];
    double*       CFD_RESTRICT g_ptrs[kMaxBatchVariables];
    for (std::size_t v = 0; v < n_vars; ++v) {
        s_ptrs[v] = s[v];
        g_ptrs[v] = g[v];
        std::fill_n(g_ptrs[v], 3 * stride, 0.0);
    }

    const auto n_own         = static_cast<std::size_t>(mesh.n_own);
    const auto n_total       = static_cast<std::size_t>(mesh.n_cells);
    const auto n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    const auto n_faces       = static_cast<std::size_t>(mesh.n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT face_area     = mesh.face_area.data();
    const double* CFD_RESTRICT face_normal_x = mesh.face_normal_x.data();
    const double* CFD_RESTRICT face_normal_y = mesh.face_normal_y.data();
    const double* CFD_RESTRICT face_normal_z = mesh.face_normal_z.data();
    const double* CFD_RESTRICT cell_volume   = mesh.cell_volume.data();

    // 1. Inner faces batch processing
    for (std::size_t face_idx = 0; face_idx < n_inner_faces; ++face_idx) {
        const auto owner = static_cast<std::size_t>(face_owner[face_idx]);
        const auto neigh = static_cast<std::size_t>(face_neigh[face_idx]);

        const double area = face_area[face_idx];
        const double nx   = face_normal_x[face_idx];
        const double ny   = face_normal_y[face_idx];
        const double nz   = face_normal_z[face_idx];

        for (std::size_t v = 0; v < n_vars; ++v) {
            const double fi_face = 0.5 * (s_ptrs[v][owner] + s_ptrs[v][neigh]);
            const double fx = area * fi_face * nx;
            const double fy = area * fi_face * ny;
            const double fz = area * fi_face * nz;

            g_ptrs[v][owner]              += fx;
            g_ptrs[v][owner + stride]     += fy;
            g_ptrs[v][owner + 2 * stride] += fz;

            g_ptrs[v][neigh]              -= fx;
            g_ptrs[v][neigh + stride]     -= fy;
            g_ptrs[v][neigh + 2 * stride] -= fz;
        }
    }

    // 2. Boundary faces batch processing
    for (std::size_t face_idx = n_inner_faces; face_idx < n_faces; ++face_idx) {
        const auto owner = static_cast<std::size_t>(face_owner[face_idx]);
        const std::size_t b_idx = n_total + face_idx - n_inner_faces;

        const double area = face_area[face_idx];
        const double nx   = face_normal_x[face_idx];
        const double ny   = face_normal_y[face_idx];
        const double nz   = face_normal_z[face_idx];

        for (std::size_t v = 0; v < n_vars; ++v) {
            const double fi_face = 0.5 * (s_ptrs[v][owner] + s_ptrs[v][b_idx]);
            const double fx = area * fi_face * nx;
            const double fy = area * fi_face * ny;
            const double fz = area * fi_face * nz;

            g_ptrs[v][owner]              += fx;
            g_ptrs[v][owner + stride]     += fy;
            g_ptrs[v][owner + 2 * stride] += fz;
        }
    }

    // 3. Normalize owned cells by volume
    for (std::size_t i = 0; i < n_own; ++i) {
        const double cv_inv = 1.0 / cell_volume[i];
        for (std::size_t v = 0; v < n_vars; ++v) {
            g_ptrs[v][i]              *= cv_inv;
            g_ptrs[v][i + stride]     *= cv_inv;
            g_ptrs[v][i + 2 * stride] *= cv_inv;
        }
    }
}


// =============================================================================
// Weighted Least-Squares (LSQ)
// =============================================================================

void LeastSquaresCellFaceGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    if (!aux_conn.has_cell_cells_face()) {
        build_cell_cells_face_conn(mesh, aux_conn.cell_cells_face_offsets, aux_conn.cell_cells_face);
        aux_conn.active_mask = aux_conn.active_mask | mesh::AuxConnType::CellCellsByFace;
    }

    const auto n_own = static_cast<std::size_t>(mesh.n_own);
    m_coeffs_off_    = aux_conn.cell_cells_face.size();
    m_coeffs_.assign(3 * m_coeffs_off_, 0.0);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_face_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_face.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;

    constexpr double eps = 1.0e-14;


    for (std::size_t c = 0; c < n_own; ++c) {
        const double x0 = cx[c];
        const double y0 = cy[c];
        const double z0 = cz[c];

        const auto pos1 = static_cast<std::size_t>(offsets[c]);
        const auto pos2 = static_cast<std::size_t>(offsets[c + 1]);
        const std::size_t n_nbs = pos2 - pos1;

        if (n_nbs == 0) continue;

        double dr_buf[kMaxCellFaceNeighbors][3];
        double w_buf[kMaxCellFaceNeighbors];

        double A[9] = {0.0};

        for (std::size_t i = 0; i < n_nbs; ++i) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[pos1 + i]);

            const double dx = cx[nb_idx] - x0;
            const double dy = cy[nb_idx] - y0;
            const double dz = cz[nb_idx] - z0;

            const double r2 = dx * dx + dy * dy + dz * dz;
            const double w  = 1.0 / std::max(r2, 1.0e-30);

            dr_buf[i][0] = dx;
            dr_buf[i][1] = dy;
            dr_buf[i][2] = dz;
            w_buf[i]     = w;

            A[0] += w * dx * dx;
            A[1] += w * dx * dy;
            A[2] += w * dx * dz;

            A[4] += w * dy * dy;
            A[5] += w * dy * dz;

            A[8] += w * dz * dz;
        }

        A[3] = A[1];
        A[6] = A[2];
        A[7] = A[5];

        // Diagonal regularization against ill-conditioned stencils
        A[0] += eps;
        A[4] += eps;
        A[8] += eps;

        double invA[9];
        invert3x3(A, invA);

        // Precompute Cx, Cy, Cz = invA * (w * dr)
        for (std::size_t i = 0; i < n_nbs; ++i) {
            const double w  = w_buf[i];
            const double rx = w * dr_buf[i][0];
            const double ry = w * dr_buf[i][1];
            const double rz = w * dr_buf[i][2];

            const std::size_t store_idx = pos1 + i;
            wx_ptr[store_idx] = invA[0] * rx + invA[1] * ry + invA[2] * rz;
            wy_ptr[store_idx] = invA[3] * rx + invA[4] * ry + invA[5] * rz;
            wz_ptr[store_idx] = invA[6] * rx + invA[7] * ry + invA[8] * rz;
        }
    }
}

void LeastSquaresCellFaceGradient::apply(const double* const CFD_RESTRICT s, 
                                         double* const CFD_RESTRICT g, 
                                         const std::size_t stride, 
                                         const mesh::MeshPart& mesh, 
                                         const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_cells_face() && "LSQ requires CellCellsByFace connectivity!");

    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_face_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_face.data();

    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;


    for (std::size_t idx = 0; idx < n_own; ++idx) {
        const double fi_self = s[idx];
        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        double gx = 0.0, gy = 0.0, gz = 0.0;

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);
            const double delta_fi = s[nb_idx] - fi_self;

            gx += wx_ptr[j] * delta_fi;
            gy += wy_ptr[j] * delta_fi;
            gz += wz_ptr[j] * delta_fi;
        }

        g[idx]              = gx;
        g[idx + stride]     = gy;
        g[idx + 2 * stride] = gz;
    }
}

void LeastSquaresCellFaceGradient::apply_set(std::span<const double*> s, 
                                             std::span<double*> g, 
                                             const std::size_t stride, 
                                             const mesh::MeshPart& mesh, 
                                             const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_cells_face() && "LSQ requires CellCellsByFace connectivity!");
    const std::size_t n_vars = s.size();
    assert(n_vars == g.size() && "Mismatched span sizes in apply_set!");
    if (n_vars == 0) return;
    assert(n_vars <= kMaxBatchVariables && "Batch size exceeds stack capacity!");

    const double* CFD_RESTRICT s_ptrs[kMaxBatchVariables];
    double*       CFD_RESTRICT g_ptrs[kMaxBatchVariables];
    for (std::size_t v = 0; v < n_vars; ++v) {
        s_ptrs[v] = s[v];
        g_ptrs[v] = g[v];
    }

    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_face_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_face.data();

    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;


    for (std::size_t idx = 0; idx < n_own; ++idx) {
        double fi_self[kMaxBatchVariables];
        double gx[kMaxBatchVariables] = {0.0};
        double gy[kMaxBatchVariables] = {0.0};
        double gz[kMaxBatchVariables] = {0.0};

        for (std::size_t v = 0; v < n_vars; ++v) {
            fi_self[v] = s_ptrs[v][idx];
        }

        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);
            const double wx   = wx_ptr[j];
            const double wy   = wy_ptr[j];
            const double wz   = wz_ptr[j];

            for (std::size_t v = 0; v < n_vars; ++v) {
                const double delta_fi = s_ptrs[v][nb_idx] - fi_self[v];
                gx[v] += wx * delta_fi;
                gy[v] += wy * delta_fi;
                gz[v] += wz * delta_fi;
            }
        }

        for (std::size_t v = 0; v < n_vars; ++v) {
            g_ptrs[v][idx]              = gx[v];
            g_ptrs[v][idx + stride]     = gy[v];
            g_ptrs[v][idx + 2 * stride] = gz[v];
        }
    }
}


// =============================================================================
// Weighted Least-Squares Cell-Node Gradient (Node-Sharing Stencil)
// =============================================================================

void LeastSquaresCellNodeGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    // 1. Build Cell -> Cells (by node) connectivity if not present
    if (!aux_conn.has_cell_cells_node()) {
        build_cell_cells_node_conn(mesh, aux_conn.cell_cells_node_offsets, aux_conn.cell_cells_node);
        aux_conn.active_mask = aux_conn.active_mask | mesh::AuxConnType::CellCellsByNode;
    }

    const auto n_own = static_cast<std::size_t>(mesh.n_own);
    m_coeffs_off_    = aux_conn.cell_cells_node.size();
    m_coeffs_.assign(3 * m_coeffs_off_, 0.0);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_node_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_node.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;

    constexpr double eps = 1.0e-14;


    for (std::size_t c = 0; c < n_own; ++c) {
        const double x0 = cx[c];
        const double y0 = cy[c];
        const double z0 = cz[c];

        const auto pos1 = static_cast<std::size_t>(offsets[c]);
        const auto pos2 = static_cast<std::size_t>(offsets[c + 1]);
        const std::size_t n_nbs = pos2 - pos1;

        if (n_nbs == 0) continue;

        double dr_buf[kMaxCellNodeNeighbors][3];
        double w_buf[kMaxCellNodeNeighbors];

        double A[9] = {0.0};

        for (std::size_t i = 0; i < n_nbs; ++i) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[pos1 + i]);

            const double dx = cx[nb_idx] - x0;
            const double dy = cy[nb_idx] - y0;
            const double dz = cz[nb_idx] - z0;

            const double r2 = dx * dx + dy * dy + dz * dz;
            const double w  = 1.0 / std::max(r2, 1.0e-30);

            dr_buf[i][0] = dx;
            dr_buf[i][1] = dy;
            dr_buf[i][2] = dz;
            w_buf[i]     = w;

            A[0] += w * dx * dx;
            A[1] += w * dx * dy;
            A[2] += w * dx * dz;

            A[4] += w * dy * dy;
            A[5] += w * dy * dz;

            A[8] += w * dz * dz;
        }

        A[3] = A[1];
        A[6] = A[2];
        A[7] = A[5];

        // Diagonal regularization
        A[0] += eps;
        A[4] += eps;
        A[8] += eps;

        double invA[9];
        invert3x3(A, invA);

        // Precompute Cx, Cy, Cz = invA * (w * dr)
        for (std::size_t i = 0; i < n_nbs; ++i) {
            const double w  = w_buf[i];
            const double rx = w * dr_buf[i][0];
            const double ry = w * dr_buf[i][1];
            const double rz = w * dr_buf[i][2];

            const std::size_t store_idx = pos1 + i;
            wx_ptr[store_idx] = invA[0] * rx + invA[1] * ry + invA[2] * rz;
            wy_ptr[store_idx] = invA[3] * rx + invA[4] * ry + invA[5] * rz;
            wz_ptr[store_idx] = invA[6] * rx + invA[7] * ry + invA[8] * rz;
        }
    }
}

void LeastSquaresCellNodeGradient::apply(const double* const CFD_RESTRICT s, 
                                         double* const CFD_RESTRICT g, 
                                         const std::size_t stride, 
                                         const mesh::MeshPart& mesh, 
                                         const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_cells_node() && "LSQ Node requires CellCellsByNode connectivity!");

    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_node_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_node.data();

    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;


    for (std::size_t idx = 0; idx < n_own; ++idx) {
        const double fi_self = s[idx];
        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        double gx = 0.0, gy = 0.0, gz = 0.0;

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);
            const double delta_fi = s[nb_idx] - fi_self;

            gx += wx_ptr[j] * delta_fi;
            gy += wy_ptr[j] * delta_fi;
            gz += wz_ptr[j] * delta_fi;
        }

        g[idx]              = gx;
        g[idx + stride]     = gy;
        g[idx + 2 * stride] = gz;
    }
}

void LeastSquaresCellNodeGradient::apply_set(std::span<const double*> s, 
                                             std::span<double*> g, 
                                             const std::size_t stride, 
                                             const mesh::MeshPart& mesh, 
                                             const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(aux_conn.has_cell_cells_node() && "LSQ Node requires CellCellsByNode connectivity!");
    const std::size_t n_vars = s.size();
    assert(n_vars == g.size() && "Mismatched span sizes in apply_set!");
    if (n_vars == 0) return;
    assert(n_vars <= kMaxBatchVariables && "Batch size exceeds stack capacity!");

    const double* CFD_RESTRICT s_ptrs[kMaxBatchVariables];
    double*       CFD_RESTRICT g_ptrs[kMaxBatchVariables];
    for (std::size_t v = 0; v < n_vars; ++v) {
        s_ptrs[v] = s[v];
        g_ptrs[v] = g[v];
    }

    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_node_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_node.data();

    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;


    for (std::size_t idx = 0; idx < n_own; ++idx) {
        double fi_self[kMaxBatchVariables];
        double gx[kMaxBatchVariables] = {0.0};
        double gy[kMaxBatchVariables] = {0.0};
        double gz[kMaxBatchVariables] = {0.0};

        for (std::size_t v = 0; v < n_vars; ++v) {
            fi_self[v] = s_ptrs[v][idx];
        }

        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);
            const double wx   = wx_ptr[j];
            const double wy   = wy_ptr[j];
            const double wz   = wz_ptr[j];

            for (std::size_t v = 0; v < n_vars; ++v) {
                const double delta_fi = s_ptrs[v][nb_idx] - fi_self[v];
                gx[v] += wx * delta_fi;
                gy[v] += wy * delta_fi;
                gz[v] += wz * delta_fi;
            }
        }

        for (std::size_t v = 0; v < n_vars; ++v) {
            g_ptrs[v][idx]              = gx[v];
            g_ptrs[v][idx + stride]     = gy[v];
            g_ptrs[v][idx + 2 * stride] = gz[v];
        }
    }
}
} // namespace cfd::solver::gradient
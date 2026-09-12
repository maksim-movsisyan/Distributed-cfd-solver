#include "cfd/numerics/gradient/gradient_operator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/localmesh.hpp"



namespace cfd::numerics::gradient {

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

constexpr std::size_t kMaxCellFaceNeighbors = 64;
constexpr std::size_t kMaxCellNodeNeighbors = 128;

} // anonymous namespace

// =============================================================================
// Green-Gauss Cell-Based (CB)
// =============================================================================

namespace {

template <std::size_t NVars>
void apply_kernel_ggcb(
    const double* CFD_RESTRICT const* CFD_RESTRICT s_ptrs,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_x,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_y,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_z,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn) noexcept {
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
        double gx[NVars] = {0.0};
        double gy[NVars] = {0.0};
        double gz[NVars] = {0.0};

        double fi_self[NVars];

        for (std::size_t v = 0; v < NVars; ++v) {
            fi_self[v] = s_ptrs[v][idx];
        }

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
            const double sign_05      = (static_cast<LocalIndex>(idx) == owner) ? 0.5 : -0.5;

            const double geom_x = sign_05 * area * nx;
            const double geom_y = sign_05 * area * ny;
            const double geom_z = sign_05 * area * nz;

            const std::size_t idx2 = (neigh >= 0)
                ? (static_cast<LocalIndex>(idx) == owner ? static_cast<std::size_t>(neigh) : static_cast<std::size_t>(owner))
                : (n_total + face_idx - n_inner_faces);

            for (std::size_t v = 0; v < NVars; ++v) {
                const double fi_face = (fi_self[v] + s_ptrs[v][idx2]);
                gx[v] += geom_x * fi_face;
                gy[v] += geom_y * fi_face;
                gz[v] += geom_z * fi_face;
            }
        }

        const double vol_inv = 1.0 / cell_volume[idx];
        for (std::size_t v = 0; v < NVars; ++v) {
            g_ptrs_x[v][idx] = gx[v] * vol_inv;
            g_ptrs_y[v][idx] = gy[v] * vol_inv;
            g_ptrs_z[v][idx] = gz[v] * vol_inv;
        }
    }
}

void apply_dynamic_ggcb(
    std::span<const double* const> s,
    std::span<double* const> gx,
    std::span<double* const> gy,
    std::span<double* const> gz,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn) noexcept {
    std::size_t v = 0;
    for (; v + 4 <= s.size(); v += 4) {
        apply_kernel_ggcb<4>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn);
    }
    for (; v < s.size(); ++v) {
        apply_kernel_ggcb<1>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn);
    }
}

} // anonymous namespace 

void GreenGaussCellGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    aux_conn.add_connectivity(mesh, mesh::AuxConnType::CellFaces);
}

void GreenGaussCellGradient::apply(const double* s, 
                                   double* CFD_RESTRICT gx,
                                   double* CFD_RESTRICT gy,
                                   double* CFD_RESTRICT gz,
                                   const mesh::MeshPart& mesh, 
                                   const mesh::MeshAuxConnectivity& aux_conn) const {
    apply_kernel_ggcb<1>(&s, &gx, &gy, &gz, mesh, aux_conn);
}

void GreenGaussCellGradient::apply_set(std::span<const double* const> s, 
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(s.size() == gx.size() && s.size() == gy.size() && s.size() == gz.size() && "Mismatched span sizes in apply_set");
    const std::size_t n_vars = s.size();

    switch (n_vars) {
        case 0: return;
        case 1: apply_kernel_ggcb<1>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break;
        case 2: apply_kernel_ggcb<2>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break;
        case 3: apply_kernel_ggcb<3>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break; // 3D Velocity
        case 4: apply_kernel_ggcb<4>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break;
        case 5: apply_kernel_ggcb<5>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break; // 3D Navier-Stokes
        case 6: apply_kernel_ggcb<6>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break; // NS + SA
        case 7: apply_kernel_ggcb<7>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn); break; // NS + k-omega SST
        default:
            apply_dynamic_ggcb(s, gx, gy, gz, mesh, aux_conn);
            break;
    }
}


// =============================================================================
// Green-Gauss Face-Based (FB)
// =============================================================================

namespace {

template <std::size_t NVars>
void apply_kernel_ggfb(
    const double* CFD_RESTRICT const* CFD_RESTRICT s_ptrs,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_x,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_y,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_z,
    const mesh::MeshPart& mesh) noexcept {
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

    for (std::size_t v = 0; v < NVars; ++v) {
        std::fill_n(g_ptrs_x[v], n_total, 0.0);
        std::fill_n(g_ptrs_y[v], n_total, 0.0);
        std::fill_n(g_ptrs_z[v], n_total, 0.0);
    }

    for (std::size_t face_idx = 0; face_idx < n_inner_faces; ++face_idx) {
        const auto owner = static_cast<std::size_t>(face_owner[face_idx]);
        const auto neigh = static_cast<std::size_t>(face_neigh[face_idx]);

        const double area_05 = 0.5 * face_area[face_idx];
        const double h_geom_x = area_05 * face_normal_x[face_idx];
        const double h_geom_y = area_05 * face_normal_y[face_idx];
        const double h_geom_z = area_05 * face_normal_z[face_idx];

        for (std::size_t v = 0; v < NVars; ++v) {
            const double fi_sum = s_ptrs[v][owner] + s_ptrs[v][neigh];
            const double fx = h_geom_x * fi_sum;
            const double fy = h_geom_y * fi_sum;
            const double fz = h_geom_z * fi_sum;

            g_ptrs_x[v][owner] += fx;
            g_ptrs_y[v][owner] += fy;
            g_ptrs_z[v][owner] += fz;

            g_ptrs_x[v][neigh] -= fx;
            g_ptrs_y[v][neigh] -= fy;
            g_ptrs_z[v][neigh] -= fz;
        }
    }

    for (std::size_t face_idx = n_inner_faces; face_idx < n_faces; ++face_idx) {
        const auto owner = static_cast<std::size_t>(face_owner[face_idx]);
        const std::size_t b_idx = n_total + face_idx - n_inner_faces;

        const double area_05 = 0.5 * face_area[face_idx];
        const double h_geom_x = area_05 * face_normal_x[face_idx];
        const double h_geom_y = area_05 * face_normal_y[face_idx];
        const double h_geom_z = area_05 * face_normal_z[face_idx];

        for (std::size_t v = 0; v < NVars; ++v) {
            const double fi_sum = s_ptrs[v][owner] + s_ptrs[v][b_idx];
            const double fx = h_geom_x * fi_sum;
            const double fy = h_geom_y * fi_sum;
            const double fz = h_geom_z * fi_sum;

            g_ptrs_x[v][owner] += fx;
            g_ptrs_y[v][owner] += fy;
            g_ptrs_z[v][owner] += fz;
        }
    }

    for (std::size_t i = 0; i < n_own; ++i) {
        const double cv_inv = 1.0 / cell_volume[i];

        for (std::size_t v = 0; v < NVars; ++v) {
            g_ptrs_x[v][i] *= cv_inv;
            g_ptrs_y[v][i] *= cv_inv;
            g_ptrs_z[v][i] *= cv_inv;
        }
    }
}

void apply_dynamic_ggfb(
    std::span<const double* const> s,
    std::span<double* const> gx,
    std::span<double* const> gy,
    std::span<double* const> gz,
    const mesh::MeshPart& mesh) noexcept {
    std::size_t v = 0;
    for (; v + 4 <= s.size(); v += 4) {
        apply_kernel_ggfb<4>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh);
    }
    for (; v < s.size(); ++v) {
        apply_kernel_ggfb<1>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh);
    }
}

} // anonymous namespace

void GreenGaussFaceGradient::setup(const mesh::MeshPart&, mesh::MeshAuxConnectivity&) {}

void GreenGaussFaceGradient::apply(const double* s, 
                                   double* CFD_RESTRICT gx,
                                   double* CFD_RESTRICT gy,
                                   double* CFD_RESTRICT gz,
                                   const mesh::MeshPart& mesh, 
                                   const mesh::MeshAuxConnectivity&) const {
    apply_kernel_ggfb<1>(&s, &gx, &gy, &gz, mesh);
}

void GreenGaussFaceGradient::apply_set(std::span<const double* const> s, 
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity&) const {
    assert(s.size() == gx.size() && s.size() == gy.size() && s.size() == gz.size() && "Mismatched span sizes in apply_set");
    const std::size_t n_vars = s.size();

    switch (n_vars) {
        case 0: return;
        case 1: apply_kernel_ggfb<1>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break;
        case 2: apply_kernel_ggfb<2>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break;
        case 3: apply_kernel_ggfb<3>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break; // 3D Velocity
        case 4: apply_kernel_ggfb<4>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break;
        case 5: apply_kernel_ggfb<5>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break; // 3D Navier-Stokes
        case 6: apply_kernel_ggfb<6>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break; // NS + SA
        case 7: apply_kernel_ggfb<7>(s.data(), gx.data(), gy.data(), gz.data(), mesh); break; // NS + k-omega SST
        default:
            apply_dynamic_ggfb(s, gx, gy, gz, mesh);
            break;
    }
}


// =============================================================================
// Weighted Least-Squares (LSQ)
// =============================================================================

namespace {

template <std::size_t NVars>
void apply_kernel_lsqf(
    const double* CFD_RESTRICT const* CFD_RESTRICT s_ptrs,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_x,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_y,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_z,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn,
    const double* CFD_RESTRICT wx_ptr,
    const double* CFD_RESTRICT wy_ptr,
    const double* CFD_RESTRICT wz_ptr) noexcept {
    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_face_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_face.data();

    for (std::size_t idx = 0; idx < n_own; ++idx) {
        double gx[NVars] = {0.0};
        double gy[NVars] = {0.0};
        double gz[NVars] = {0.0};

        double fi_self[NVars];

        for (std::size_t v = 0; v < NVars; ++v) {
            fi_self[v] = s_ptrs[v][idx];
        }

        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);

            const double wx = wx_ptr[j];
            const double wy = wy_ptr[j];
            const double wz = wz_ptr[j];


            for (std::size_t v = 0; v < NVars; ++v) {
                const double delta_fi = s_ptrs[v][nb_idx] - fi_self[v];
                gx[v] += wx * delta_fi;
                gy[v] += wy * delta_fi;
                gz[v] += wz * delta_fi;
            }
        }

        for (std::size_t v = 0; v < NVars; ++v) {
            g_ptrs_x[v][idx] = gx[v];
            g_ptrs_y[v][idx] = gy[v];
            g_ptrs_z[v][idx] = gz[v];
        }
    }
}

void apply_dynamic_lsqf(
    std::span<const double* const> s,
    std::span<double* const> gx,
    std::span<double* const> gy,
    std::span<double* const> gz,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn,
    const double* CFD_RESTRICT wx_ptr,
    const double* CFD_RESTRICT wy_ptr,
    const double* CFD_RESTRICT wz_ptr) noexcept {
    std::size_t v = 0;
    for (; v + 4 <= s.size(); v += 4) {
        apply_kernel_lsqf<4>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
    }
    for (; v < s.size(); ++v) {
        apply_kernel_lsqf<1>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
    }
}

} // anonymous namespace

void LeastSquaresCellFaceGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    aux_conn.add_connectivity(mesh, mesh::AuxConnType::CellCellsByFace);
    
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

void LeastSquaresCellFaceGradient::apply(const double* s, 
                                   double* CFD_RESTRICT gx,
                                   double* CFD_RESTRICT gy,
                                   double* CFD_RESTRICT gz,
                                   const mesh::MeshPart& mesh, 
                                   const mesh::MeshAuxConnectivity& aux_conn) const {
    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;
    apply_kernel_lsqf<1>(&s, &gx, &gy, &gz, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
}

void LeastSquaresCellFaceGradient::apply_set(std::span<const double* const> s, 
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(s.size() == gx.size() && s.size() == gy.size() && s.size() == gz.size() && "Mismatched span sizes in apply_set");
    const std::size_t n_vars = s.size();
    
    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;

    switch (n_vars) {
        case 0: return;
        case 1: apply_kernel_lsqf<1>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 2: apply_kernel_lsqf<2>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 3: apply_kernel_lsqf<3>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // 3D Velocity
        case 4: apply_kernel_lsqf<4>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 5: apply_kernel_lsqf<5>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // 3D Navier-Stokes
        case 6: apply_kernel_lsqf<6>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // NS + SA
        case 7: apply_kernel_lsqf<7>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // NS + k-omega SST
        default:
            apply_dynamic_lsqf(s, gx, gy, gz, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
            break;
    }
}


// =============================================================================
// Weighted Least-Squares Cell-Node Gradient (Node-Sharing Stencil)
// =============================================================================

namespace {

template <std::size_t NVars>
void apply_kernel_lsqn(
    const double* CFD_RESTRICT const* CFD_RESTRICT s_ptrs,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_x,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_y,
    double* CFD_RESTRICT const* CFD_RESTRICT g_ptrs_z,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn,
    const double* CFD_RESTRICT wx_ptr,
    const double* CFD_RESTRICT wy_ptr,
    const double* CFD_RESTRICT wz_ptr) noexcept {
    const auto n_own = static_cast<std::size_t>(mesh.n_own);

    const LocalIndex* CFD_RESTRICT offsets   = aux_conn.cell_cells_node_offsets.data();
    const LocalIndex* CFD_RESTRICT neighbors = aux_conn.cell_cells_node.data();

    for (std::size_t idx = 0; idx < n_own; ++idx) {
        double gx[NVars] = {0.0};
        double gy[NVars] = {0.0};
        double gz[NVars] = {0.0};

        double fi_self[NVars];

        for (std::size_t v = 0; v < NVars; ++v) {
            fi_self[v] = s_ptrs[v][idx];
        }

        const auto pos1 = static_cast<std::size_t>(offsets[idx]);
        const auto pos2 = static_cast<std::size_t>(offsets[idx + 1]);

        for (std::size_t j = pos1; j < pos2; ++j) {
            const auto nb_idx = static_cast<std::size_t>(neighbors[j]);

            const double wx = wx_ptr[j];
            const double wy = wy_ptr[j];
            const double wz = wz_ptr[j];

            for (std::size_t v = 0; v < NVars; ++v) {
                const double delta_fi = s_ptrs[v][nb_idx] - fi_self[v];
                gx[v] += wx * delta_fi;
                gy[v] += wy * delta_fi;
                gz[v] += wz * delta_fi;
            }
        }

        for (std::size_t v = 0; v < NVars; ++v) {
            g_ptrs_x[v][idx] = gx[v];
            g_ptrs_y[v][idx] = gy[v];
            g_ptrs_z[v][idx] = gz[v];
        }
    }
}

void apply_dynamic_lsqn(
    std::span<const double* const> s,
    std::span<double* const> gx,
    std::span<double* const> gy,
    std::span<double* const> gz,
    const mesh::MeshPart& mesh,
    const mesh::MeshAuxConnectivity& aux_conn,
    const double* CFD_RESTRICT wx_ptr,
    const double* CFD_RESTRICT wy_ptr,
    const double* CFD_RESTRICT wz_ptr) noexcept {
    std::size_t v = 0;
    for (; v + 4 <= s.size(); v += 4) {
        apply_kernel_lsqn<4>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
    }
    for (; v < s.size(); ++v) {
        apply_kernel_lsqn<1>(s.data() + v, gx.data() + v, gy.data() + v, gz.data() + v, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
    }
}

} // anonymous namespace

void LeastSquaresCellNodeGradient::setup(const mesh::MeshPart& mesh, mesh::MeshAuxConnectivity& aux_conn) {
    aux_conn.add_connectivity(mesh, mesh::AuxConnType::CellCellsByNode);
    
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

void LeastSquaresCellNodeGradient::apply(const double* s, 
                                   double* CFD_RESTRICT gx,
                                   double* CFD_RESTRICT gy,
                                   double* CFD_RESTRICT gz,
                                   const mesh::MeshPart& mesh, 
                                   const mesh::MeshAuxConnectivity& aux_conn) const {
    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;
    apply_kernel_lsqn<1>(&s, &gx, &gy, &gz, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
}

void LeastSquaresCellNodeGradient::apply_set(std::span<const double* const> s, 
                                       std::span<double* const> gx,
                                       std::span<double* const> gy,
                                       std::span<double* const> gz,
                                       const mesh::MeshPart& mesh, 
                                       const mesh::MeshAuxConnectivity& aux_conn) const {
    assert(s.size() == gx.size() && s.size() == gy.size() && s.size() == gz.size() && "Mismatched span sizes in apply_set");
    const std::size_t n_vars = s.size();
    
    const double* CFD_RESTRICT wx_ptr = m_coeffs_.data();
    const double* CFD_RESTRICT wy_ptr = m_coeffs_.data() + m_coeffs_off_;
    const double* CFD_RESTRICT wz_ptr = m_coeffs_.data() + 2 * m_coeffs_off_;

    switch (n_vars) {
        case 0: return;
        case 1: apply_kernel_lsqn<1>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 2: apply_kernel_lsqn<2>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 3: apply_kernel_lsqn<3>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // 3D Velocity
        case 4: apply_kernel_lsqn<4>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break;
        case 5: apply_kernel_lsqn<5>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // 3D Navier-Stokes
        case 6: apply_kernel_lsqn<6>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // NS + SA
        case 7: apply_kernel_lsqn<7>(s.data(), gx.data(), gy.data(), gz.data(), mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr); break; // NS + k-omega SST
        default:
            apply_dynamic_lsqn(s, gx, gy, gz, mesh, aux_conn, wx_ptr, wy_ptr, wz_ptr);
            break;
    }
}

} // namespace cfd::numerics::gradient
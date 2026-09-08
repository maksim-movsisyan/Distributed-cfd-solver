// MUSCL 2nd-order spatial reconstruction on PRIMITIVE variables,
// templated on a gradient limiter policy (limiter/limiters.hpp).
//
// Face states:
//   qL = q(c0) + phi(c0) * (grad q(c0) . d0),  d0 = x_face - x_c0
//   qR = q(c1) + phi(c1) * (grad q(c1) . d1),  d1 = x_face - x_c1
//
// Boundary faces use the BC ghost value directly as qR.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/limiter/limiters.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver::recon {

template <typename LimiterPolicy>
struct Muscl {
    static constexpr bool kNeedsGradients = true;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::CellFaces|
                                                          mesh::AuxConnType::CellCellsByFace;
    static constexpr const char* name() noexcept { return "MUSCL"; }
    static constexpr const char* limiter_name() noexcept { return LimiterPolicy::name(); }

    /**
     * @brief Per-stage limiter evaluation over owned cells.
     */
    static inline void compute_limiters(const mesh::MeshPart& mesh,
                                 const mesh::MeshAuxConnectivity& aux_conn,
                                 fields::ConstPrimitiveView q,
                                 fields::ConstPrimitiveGradView grad,
                                 fields::PrimitiveView<double> phi,
                                 const double venkat_k = 1.0) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mesh.n_own);

        const LocalIndex* CFD_RESTRICT c2c_off = aux_conn.cell_cells_face_offsets.data();
        const LocalIndex* CFD_RESTRICT c2c_val = aux_conn.cell_cells_face.data();
        const LocalIndex* CFD_RESTRICT c2f_off = aux_conn.cell_faces_offsets.data();
        const LocalIndex* CFD_RESTRICT c2f_val = aux_conn.cell_faces.data();

        const double* CFD_RESTRICT prs = q.prs;
        const double* CFD_RESTRICT vx  = q.vx;
        const double* CFD_RESTRICT vy  = q.vy;
        const double* CFD_RESTRICT vz  = q.vz;
        const double* CFD_RESTRICT tmp = q.tmp;

        const double* CFD_RESTRICT face_centroid_x = mesh.face_centroid_x.data();
        const double* CFD_RESTRICT face_centroid_y = mesh.face_centroid_y.data();
        const double* CFD_RESTRICT face_centroid_z = mesh.face_centroid_z.data();
        const double* CFD_RESTRICT cell_centroid_x = mesh.cell_centroid_x.data();
        const double* CFD_RESTRICT cell_centroid_y = mesh.cell_centroid_y.data();
        const double* CFD_RESTRICT cell_centroid_z = mesh.cell_centroid_z.data();

        // loop over all owned cells
        for (std::size_t c = 0; c < n_own; ++c) {
            // 1. Determine neighbor extrema (including self)
            double qmax[constants::kNumVars] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};
            double qmin[constants::kNumVars] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};
            const double qc[constants::kNumVars] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};
            
            // loop over all cell neighbors
            for (LocalIndex j = c2c_off[c]; j < c2c_off[c + 1]; ++j) {
                const std::size_t js = static_cast<std::size_t>(c2c_val[j]);
                qmax[0] = std::max(qmax[0], prs[js]);
                qmin[0] = std::min(qmin[0], prs[js]);

                qmax[1] = std::max(qmax[1], vx[js]);
                qmin[1] = std::min(qmin[1], vx[js]);

                qmax[2] = std::max(qmax[2], vy[js]);
                qmin[2] = std::min(qmin[2], vy[js]);

                qmax[3] = std::max(qmax[3], vz[js]);
                qmin[3] = std::min(qmin[3], vz[js]);

                qmax[4] = std::max(qmax[4], tmp[js]);
                qmin[4] = std::min(qmin[4], tmp[js]);
            } // end loop over all cell neighbors

            // 2. Scan cell faces and accumulate running minimum of limiter
            double ph[constants::kNumVars] = {1.0, 1.0, 1.0, 1.0, 1.0};
            const double k3 = venkat_k * venkat_k * venkat_k;
            const double eps2 = k3 * mesh.cell_volume[c];

            const double g_p[3]  = {grad.dprs_dx(c), grad.dprs_dy(c), grad.dprs_dz(c)};
            const double g_vx[3] = {grad.dvx_dx(c),  grad.dvx_dy(c),  grad.dvx_dz(c)};
            const double g_vy[3] = {grad.dvy_dx(c),  grad.dvy_dy(c),  grad.dvy_dz(c)};
            const double g_vz[3] = {grad.dvz_dx(c),  grad.dvz_dy(c),  grad.dvz_dz(c)};
            const double g_t[3]  = {grad.dtmp_dx(c), grad.dtmp_dy(c), grad.dtmp_dz(c)};

            // loop over all cell faces
            for (LocalIndex e = c2f_off[c]; e < c2f_off[c + 1]; ++e) {
                const std::size_t es = static_cast<std::size_t>(e);
                const std::size_t f = static_cast<std::size_t>(c2f_val[es]);

                const double ex = face_centroid_x[f] - cell_centroid_x[c];
                const double ey = face_centroid_y[f] - cell_centroid_y[c];
                const double ez = face_centroid_z[f] - cell_centroid_z[c];

                const double df[constants::kNumVars] = {
                    g_p[0]  * ex + g_p[1]  * ey + g_p[2]  * ez,
                    g_vx[0] * ex + g_vx[1] * ey + g_vx[2] * ez,
                    g_vy[0] * ex + g_vy[1] * ey + g_vy[2] * ez,
                    g_vz[0] * ex + g_vz[1] * ey + g_vz[2] * ez,
                    g_t[0]  * ex + g_t[1]  * ey + g_t[2]  * ez
                };

                for (std::size_t v = 0; v < constants::kNumVars; ++v) {
                    const double d_nb = (df[v] > 0.0) ? (qmax[v] - qc[v]) : (qmin[v] - qc[v]);
                    ph[v] = std::min(ph[v], LimiterPolicy::phi(d_nb, df[v], eps2));
                }
            } // end loop over all cell faces

            phi.prs[c] = ph[0];
            phi.vx[c]  = ph[1];
            phi.vy[c]  = ph[2];
            phi.vz[c]  = ph[3];
            phi.tmp[c] = ph[4];
        }
    }

    /**
     * @brief Interior face primitive reconstruction via limited gradients.
     */
    static inline void face_states(const ReconField& s,
                            const mesh::MeshPart& mesh,
                            const mesh::MeshAuxGeometry& /*aux_geom*/,
                            const std::size_t f,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[constants::kNumVars],
                            double qR[constants::kNumVars]) noexcept {
        const double d0x = mesh.face_centroid_x[f] - mesh.cell_centroid_x[c0];
        const double d0y = mesh.face_centroid_y[f] - mesh.cell_centroid_y[c0];
        const double d0z = mesh.face_centroid_z[f] - mesh.cell_centroid_z[c0];

        const double d1x = mesh.face_centroid_x[f] - mesh.cell_centroid_x[c1];
        const double d1y = mesh.face_centroid_y[f] - mesh.cell_centroid_y[c1];
        const double d1z = mesh.face_centroid_z[f] - mesh.cell_centroid_z[c1];

        // Owner cell state (L)
        qL[0] = s.q.prs[c0] + s.phi.prs[c0] * (s.grad.dprs_dx(c0) * d0x + s.grad.dprs_dy(c0) * d0y + s.grad.dprs_dz(c0) * d0z);
        qL[1] = s.q.vx[c0]  + s.phi.vx[c0]  * (s.grad.dvx_dx(c0)  * d0x + s.grad.dvx_dy(c0)  * d0y + s.grad.dvx_dz(c0)  * d0z);
        qL[2] = s.q.vy[c0]  + s.phi.vy[c0]  * (s.grad.dvy_dx(c0)  * d0x + s.grad.dvy_dy(c0)  * d0y + s.grad.dvy_dz(c0)  * d0z);
        qL[3] = s.q.vz[c0]  + s.phi.vz[c0]  * (s.grad.dvz_dx(c0)  * d0x + s.grad.dvz_dy(c0)  * d0y + s.grad.dvz_dz(c0)  * d0z);
        qL[4] = s.q.tmp[c0] + s.phi.tmp[c0] * (s.grad.dtmp_dx(c0) * d0x + s.grad.dtmp_dy(c0) * d0y + s.grad.dtmp_dz(c0) * d0z);

        // Neighbour cell state (R)
        qR[0] = s.q.prs[c1] + s.phi.prs[c1] * (s.grad.dprs_dx(c1) * d1x + s.grad.dprs_dy(c1) * d1y + s.grad.dprs_dz(c1) * d1z);
        qR[1] = s.q.vx[c1]  + s.phi.vx[c1]  * (s.grad.dvx_dx(c1)  * d1x + s.grad.dvx_dy(c1)  * d1y + s.grad.dvx_dz(c1)  * d1z);
        qR[2] = s.q.vy[c1]  + s.phi.vy[c1]  * (s.grad.dvy_dx(c1)  * d1x + s.grad.dvy_dy(c1)  * d1y + s.grad.dvy_dz(c1)  * d1z);
        qR[3] = s.q.vz[c1]  + s.phi.vz[c1]  * (s.grad.dvz_dx(c1)  * d1x + s.grad.dvz_dy(c1)  * d1y + s.grad.dvz_dz(c1)  * d1z);
        qR[4] = s.q.tmp[c1] + s.phi.tmp[c1] * (s.grad.dtmp_dx(c1) * d1x + s.grad.dtmp_dy(c1) * d1y + s.grad.dtmp_dz(c1) * d1z);
    }

    /**
     * @brief Boundary face reconstruction.
     * The ghost state cg carries the exact BC state, owner c0 uses its cell centroid state.
     */
    static inline void boundary_face_states(const ReconField& s,
                                     const mesh::MeshPart& /*mesh*/,
                                     const mesh::MeshAuxGeometry& /*aux_geom*/,
                                     const std::size_t /*f*/,
                                     const std::size_t c0,
                                     const std::size_t cg,
                                     double qL[constants::kNumVars],
                                     double qR[constants::kNumVars]) noexcept {
        qL[0] = s.q.prs[c0];
        qL[1] = s.q.vx[c0];
        qL[2] = s.q.vy[c0];
        qL[3] = s.q.vz[c0];
        qL[4] = s.q.tmp[c0];

        qR[0] = s.q.prs[cg];
        qR[1] = s.q.vx[cg];
        qR[2] = s.q.vy[cg];
        qR[3] = s.q.vz[cg];
        qR[4] = s.q.tmp[cg];
    }
};

static_assert(ReconstructionPolicy<Muscl<limiter::BarthJespersen>>);
static_assert(ReconstructionPolicy<Muscl<limiter::Venkatakrishnan>>);
static_assert(ReconstructionPolicy<Muscl<limiter::VanAlbada>>);

} // namespace cfd::solver::recon
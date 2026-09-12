#pragma once

#include <algorithm>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/numerics/limiter/slope_limiters.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"

namespace cfd::numerics::recon {

template <typename LimiterPolicy>
struct Muscl {
    static constexpr bool kNeedsGradients = true;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr mesh::AuxConnType kAuxConnectivity = 
        mesh::AuxConnType::CellFaces | mesh::AuxConnType::CellCellsByFace;

    static constexpr const char* name() noexcept { return "MUSCL"; }
    static constexpr const char* limiter_name() noexcept { return LimiterPolicy::name(); }

    template <std::size_t NVars>
    static void compute_limiters(
        const mesh::MeshPart& mesh,
        const mesh::MeshAuxConnectivity& aux_conn,
        const double* CFD_RESTRICT const* CFD_RESTRICT q,
        const double* CFD_RESTRICT const* CFD_RESTRICT grad_x,
        const double* CFD_RESTRICT const* CFD_RESTRICT grad_y,
        const double* CFD_RESTRICT const* CFD_RESTRICT grad_z,
        double* CFD_RESTRICT const* CFD_RESTRICT phi,
        const double venkat_k = 1.0) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mesh.n_own);

        const LocalIndex* CFD_RESTRICT c2c_off = aux_conn.cell_cells_face_offsets.data();
        const LocalIndex* CFD_RESTRICT c2c_val = aux_conn.cell_cells_face.data();
        const LocalIndex* CFD_RESTRICT c2f_off = aux_conn.cell_faces_offsets.data();
        const LocalIndex* CFD_RESTRICT c2f_val = aux_conn.cell_faces.data();

        const double* CFD_RESTRICT face_centroid_x = mesh.face_centroid_x.data();
        const double* CFD_RESTRICT face_centroid_y = mesh.face_centroid_y.data();
        const double* CFD_RESTRICT face_centroid_z = mesh.face_centroid_z.data();
        const double* CFD_RESTRICT cell_centroid_x = mesh.cell_centroid_x.data();
        const double* CFD_RESTRICT cell_centroid_y = mesh.cell_centroid_y.data();
        const double* CFD_RESTRICT cell_centroid_z = mesh.cell_centroid_z.data();

        const double k3 = venkat_k * venkat_k * venkat_k;

        // loop over all local cells
        for (std::size_t c = 0; c < n_own; ++c) {
            double qmax[NVars];
            double qmin[NVars];
            double qc[NVars];

            for (std::size_t v = 0; v < NVars; ++v) {
                qc[v]   = q[v][c];
                qmax[v] = qc[v];
                qmin[v] = qc[v];
            }

            // 1. Finding local extrema among neigbhors
            for (LocalIndex j = c2c_off[c]; j < c2c_off[c + 1]; ++j) {
                const std::size_t js = static_cast<std::size_t>(c2c_val[j]);

                for (std::size_t v = 0; v < NVars; ++v) {
                    const double q_nb = q[v][js];
                    qmax[v] = std::max(qmax[v], q_nb);
                    qmin[v] = std::min(qmin[v], q_nb);
                }
            }

            // 2. Scan cell faces and accumulate running minimum of limiter
            double ph[NVars];
            for (std::size_t v = 0; v < NVars; ++v) { ph[v] = 1.0; }

            const double eps2 = k3 * mesh.cell_volume[c];
            const double cc_x = cell_centroid_x[c];
            const double cc_y = cell_centroid_y[c];
            const double cc_z = cell_centroid_z[c];

            for (LocalIndex e = c2f_off[c]; e < c2f_off[c + 1]; ++e) {
                const std::size_t es = static_cast<std::size_t>(e);
                const std::size_t f = static_cast<std::size_t>(c2f_val[es]);

                const double ex = face_centroid_x[f] - cc_x;
                const double ey = face_centroid_y[f] - cc_y;
                const double ez = face_centroid_z[f] - cc_z;

                for (std::size_t v = 0; v < NVars; ++v) {
                    const double df = grad_x[v][c] * ex +
                                      grad_y[v][c] * ey +
                                      grad_z[v][c] * ez;

                    const double d_nb = (df > 0.0) ? (qmax[v] - qc[v]) : (qmin[v] - qc[v]);
                    ph[v] = std::min(ph[v], LimiterPolicy::phi(d_nb, df, eps2));
                }
            }

            for (std::size_t v = 0; v < NVars; ++v) {
                phi[v][c] = ph[v];
            }
        }
    }

    template <std::size_t NVars>
    static inline void face_states(
        const ReconBatchField<NVars>& s,
        const mesh::MeshPart& mesh,
        const mesh::MeshAuxGeometry& /*aux_geom*/,
        const std::size_t f,
        const std::size_t c0,
        const std::size_t c1,
        double* CFD_RESTRICT qL,
        double* CFD_RESTRICT qR) noexcept {
        const double fx = mesh.face_centroid_x[f];
        const double fy = mesh.face_centroid_y[f];
        const double fz = mesh.face_centroid_z[f];

        const double d0x = fx - mesh.cell_centroid_x[c0];
        const double d0y = fy - mesh.cell_centroid_y[c0];
        const double d0z = fz - mesh.cell_centroid_z[c0];

        const double d1x = fx - mesh.cell_centroid_x[c1];
        const double d1y = fy - mesh.cell_centroid_y[c1];
        const double d1z = fz - mesh.cell_centroid_z[c1];

        for (std::size_t v = 0; v < NVars; ++v) {
            const double dot0 = s.grad_x[v][c0] * d0x + 
                                s.grad_y[v][c0] * d0y + 
                                s.grad_z[v][c0] * d0z;
            qL[v] = s.q[v][c0] + s.phi[v][c0] * dot0;

            const double dot1 = s.grad_x[v][c1] * d1x + 
                                s.grad_y[v][c1] * d1y + 
                                s.grad_z[v][c1] * d1z;
            qR[v] = s.q[v][c1] + s.phi[v][c1] * dot1;
        }
    }

    template <std::size_t NVars>
    static inline void boundary_face_states(
        const ReconBatchField<NVars>& s,
        const mesh::MeshPart& /*mesh*/,
        const mesh::MeshAuxGeometry& /*aux_geom*/,
        const std::size_t /*f*/,
        const std::size_t c0,
        const std::size_t cg,
        double* CFD_RESTRICT qL,
        double* CFD_RESTRICT qR) noexcept {
        for (std::size_t v = 0; v < NVars; ++v) {
            qL[v] = s.q[v][c0];
            qR[v] = s.q[v][cg];
        }
    }
};

static_assert(ReconstructionPolicy<Muscl<numerics::limiter::BarthJespersen>, 1>);
static_assert(ReconstructionPolicy<Muscl<numerics::limiter::BarthJespersen>, 5>);
static_assert(ReconstructionPolicy<Muscl<numerics::limiter::Venkatakrishnan>, 5>);
static_assert(ReconstructionPolicy<Muscl<numerics::limiter::VanAlbada>, 5>);

} // namespace cfd::numerics::recon
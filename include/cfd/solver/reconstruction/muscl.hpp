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
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/limiter/limiters.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver::recon {

template <typename LimiterPolicy>
struct Muscl {
    static constexpr bool kNeedsGradients = true;
    static constexpr const char* name() noexcept { return "MUSCL"; }
    static constexpr const char* limiter_name() noexcept { return LimiterPolicy::name(); }

    /**
     * @struct Geometry
     * @brief Mesh-fixed precomputed data for MUSCL extrapolation and limiting.
     */
    struct Geometry {
        // Reconstruction vectors per face:
        // d0 = x_face - x_owner, d1 = x_face - x_neigh
        std::vector<double> d0x, d0y, d0z; ///< [n_faces]
        std::vector<double> d1x, d1y, d1z; ///< [n_faces]

        // Cell -> Face incidence stencil of owned cells for the limiter sweep.
        // Stores pre-directed displacement vector (cell centroid -> face centroid).
        std::vector<LocalIndex> cell_face_offsets; ///< [n_own + 1]
        std::vector<LocalIndex> cell_face_ids;     ///< [n_entries]
        std::vector<double> cell_face_dx;          ///< [n_entries]
        std::vector<double> cell_face_dy;          ///< [n_entries]
        std::vector<double> cell_face_dz;          ///< [n_entries]

        // Venkatakrishnan smoothing parameter: eps2 = (k * V^{1/3})^3 = k^3 * V
        std::vector<double> eps2; ///< [n_own]
    };

    /**
     * @brief Precomputes static geometry vectors and cell-face stencils once at startup.
     */
    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& mp,
                                                 const double venkat_k = 1.0) {
        const std::size_t n_inner = static_cast<std::size_t>(mp.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);
        const std::size_t n_own   = static_cast<std::size_t>(mp.n_own);

        Geometry g;
        g.d0x.resize(n_faces);
        g.d0y.resize(n_faces);
        g.d0z.resize(n_faces);
        g.d1x.resize(n_faces);
        g.d1y.resize(n_faces);
        g.d1z.resize(n_faces);

        // 1. Precompute face reconstruction vectors
        for (std::size_t f = 0; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);
            g.d0x[f] = mp.face_centroid_x[f] - mp.cell_centroid_x[c0];
            g.d0y[f] = mp.face_centroid_y[f] - mp.cell_centroid_y[c0];
            g.d0z[f] = mp.face_centroid_z[f] - mp.cell_centroid_z[c0];

            if (f < n_inner) {
                const std::size_t c1 = static_cast<std::size_t>(mp.face_neigh[f]);
                g.d1x[f] = mp.face_centroid_x[f] - mp.cell_centroid_x[c1];
                g.d1y[f] = mp.face_centroid_y[f] - mp.cell_centroid_y[c1];
                g.d1z[f] = mp.face_centroid_z[f] - mp.cell_centroid_z[c1];
            } else {
                g.d1x[f] = 0.0;
                g.d1y[f] = 0.0;
                g.d1z[f] = 0.0;
            }
        }

        // 2. Build Cell -> Face CSR stencil (for owned cells only)
        g.cell_face_offsets.assign(n_own + 1, 0);
        const auto count_incidence = [&](const LocalIndex c) {
            if (c >= 0 && c < static_cast<LocalIndex>(n_own)) {
                ++g.cell_face_offsets[static_cast<std::size_t>(c) + 1];
            }
        };

        for (std::size_t f = 0; f < n_faces; ++f) {
            count_incidence(mp.face_owner[f]);
            if (f < n_inner) {
                count_incidence(mp.face_neigh[f]);
            }
        }

        for (std::size_t c = 0; c < n_own; ++c) {
            g.cell_face_offsets[c + 1] += g.cell_face_offsets[c];
        }

        const std::size_t n_entries = static_cast<std::size_t>(g.cell_face_offsets.back());
        g.cell_face_ids.resize(n_entries);
        g.cell_face_dx.resize(n_entries);
        g.cell_face_dy.resize(n_entries);
        g.cell_face_dz.resize(n_entries);

        std::vector<LocalIndex> cursor(g.cell_face_offsets.begin(), g.cell_face_offsets.end() - 1);
        const auto insert_entry = [&](const std::size_t f, const LocalIndex c, const bool is_owner) {
            if (c < 0 || c >= static_cast<LocalIndex>(n_own)) {
                return;
            }
            const std::size_t cs = static_cast<std::size_t>(c);
            const std::size_t e  = static_cast<std::size_t>(cursor[cs]++);

            g.cell_face_ids[e] = static_cast<LocalIndex>(f);
            g.cell_face_dx[e]  = is_owner ? g.d0x[f] : g.d1x[f];
            g.cell_face_dy[e]  = is_owner ? g.d0y[f] : g.d1y[f];
            g.cell_face_dz[e]  = is_owner ? g.d0z[f] : g.d1z[f];
        };

        for (std::size_t f = 0; f < n_faces; ++f) {
            insert_entry(f, mp.face_owner[f], true);
            if (f < n_inner) {
                insert_entry(f, mp.face_neigh[f], false);
            }
        }

        // 3. Precompute limiter threshold eps2
        g.eps2.resize(n_own);
        const double k3 = venkat_k * venkat_k * venkat_k;
        for (std::size_t c = 0; c < n_own; ++c) {
            g.eps2[c] = k3 * mp.cell_volume[c];
        }

        return g;
    }

    /**
     * @brief Per-stage limiter evaluation over owned cells.
     */
    static void compute_limiters(const mesh::MeshPart& mp,
                                 fields::ConstPrimitiveView q,
                                 fields::ConstPrimitiveGradView grad,
                                 const gradient::VertexAdjacency& adj,
                                 const Geometry& g,
                                 fields::PrimitiveView<double> phi) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mp.n_own);

        const LocalIndex* CFD_RESTRICT nbo = adj.offsets.data();
        const LocalIndex* CFD_RESTRICT nbc = adj.cells.data();

        const double* CFD_RESTRICT prs = q.prs;
        const double* CFD_RESTRICT vx  = q.vx;
        const double* CFD_RESTRICT vy  = q.vy;
        const double* CFD_RESTRICT vz  = q.vz;
        const double* CFD_RESTRICT tmp = q.tmp;

        for (std::size_t c = 0; c < n_own; ++c) {
            // 1. Determine neighbor extrema (including self)
            double qmax[kNumPrimitives] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};
            double qmin[kNumPrimitives] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};

            for (LocalIndex j = nbo[c]; j < nbo[c + 1]; ++j) {
                const auto js = static_cast<std::size_t>(nbc[j]);
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
            }

            // 2. Scan cell faces and accumulate running minimum of limiter
            double ph[kNumPrimitives] = {1.0, 1.0, 1.0, 1.0, 1.0};
            const double eps2 = g.eps2[c];

            const double g_p[3]  = {grad.dprs_dx(c), grad.dprs_dy(c), grad.dprs_dz(c)};
            const double g_vx[3] = {grad.dvx_dx(c),  grad.dvx_dy(c),  grad.dvx_dz(c)};
            const double g_vy[3] = {grad.dvy_dx(c),  grad.dvy_dy(c),  grad.dvy_dz(c)};
            const double g_vz[3] = {grad.dvz_dx(c),  grad.dvz_dy(c),  grad.dvz_dz(c)};
            const double g_t[3]  = {grad.dtmp_dx(c), grad.dtmp_dy(c), grad.dtmp_dz(c)};

            for (LocalIndex e = g.cell_face_offsets[c]; e < g.cell_face_offsets[c + 1]; ++e) {
                const std::size_t es = static_cast<std::size_t>(e);
                const double ex = g.cell_face_dx[es];
                const double ey = g.cell_face_dy[es];
                const double ez = g.cell_face_dz[es];

                const double df[kNumPrimitives] = {
                    g_p[0]  * ex + g_p[1]  * ey + g_p[2]  * ez,
                    g_vx[0] * ex + g_vx[1] * ey + g_vx[2] * ez,
                    g_vy[0] * ex + g_vy[1] * ey + g_vy[2] * ez,
                    g_vz[0] * ex + g_vz[1] * ey + g_vz[2] * ez,
                    g_t[0]  * ex + g_t[1]  * ey + g_t[2]  * ez
                };

                const double qc[kNumPrimitives] = {prs[c], vx[c], vy[c], vz[c], tmp[c]};

                for (std::size_t v = 0; v < kNumPrimitives; ++v) {
                    const double d_nb = (df[v] > 0.0) ? (qmax[v] - qc[v]) : (qmin[v] - qc[v]);
                    ph[v] = std::min(ph[v], LimiterPolicy::phi(d_nb, df[v], eps2));
                }
            }

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
    static void face_states(const ReconField& s,
                            const Geometry& g,
                            const std::size_t f,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[kNumPrimitives],
                            double qR[kNumPrimitives]) noexcept {
        const double d0x = g.d0x[f];
        const double d0y = g.d0y[f];
        const double d0z = g.d0z[f];

        const double d1x = g.d1x[f];
        const double d1y = g.d1y[f];
        const double d1z = g.d1z[f];

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
    static void boundary_face_states(const ReconField& s,
                                     const Geometry& /*g*/,
                                     const std::size_t /*f*/,
                                     const std::size_t c0,
                                     const std::size_t cg,
                                     double qL[kNumPrimitives],
                                     double qR[kNumPrimitives]) noexcept {
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
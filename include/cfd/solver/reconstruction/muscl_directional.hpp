#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"
#include "cfd/solver/limiter/limiters.hpp"

namespace cfd::solver::recon {

// --- Directional MUSCL Reconstruction Policy ---------------------------------

template <typename Limiter1D>
struct MusclDirectional {
    static constexpr bool kNeedsGradients = true;
    static constexpr const char* name() noexcept { return "MUSCL_DIRECTIONAL"; }
    static constexpr const char* limiter_name() noexcept { return Limiter1D::name(); }

    struct Geometry {
        // Vector pointing from owner cell centroid to neighbor cell centroid:
        // xi = x_c1 - x_c0
        std::vector<double> xix, xiy, xiz; // [n_faces]
    };

    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& mp,
                                                 const double /*venkat_k*/ = 0.0) {
        const std::size_t n_inner = static_cast<std::size_t>(mp.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);

        Geometry g;
        g.xix.resize(n_faces, 0.0);
        g.xiy.resize(n_faces, 0.0);
        g.xiz.resize(n_faces, 0.0);

        for (std::size_t f = 0; f < n_inner; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(mp.face_neigh[f]);
            g.xix[f] = mp.cell_centroid_x[c1] - mp.cell_centroid_x[c0];
            g.xiy[f] = mp.cell_centroid_y[c1] - mp.cell_centroid_y[c0];
            g.xiz[f] = mp.cell_centroid_z[c1] - mp.cell_centroid_z[c0];
        }
        return g;
    }

    // No cell-wise limiter sweep needed -> zero cost!
    template <typename VertexAdj>
    static void compute_limiters(const mesh::MeshPart& /*mp*/,
                                 fields::ConstPrimitiveView /*q*/,
                                 fields::ConstPrimitiveGradView /*grad*/,
                                 const VertexAdj& /*adj*/,
                                 const Geometry& /*geom*/,
                                 fields::PrimitiveView<double> /*phi*/) noexcept {}

    static void face_states(const ReconField& s,
                            const Geometry& g,
                            const std::size_t f,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[kNumPrimitives],
                            double qR[kNumPrimitives]) noexcept {
        constexpr double kEps = 1.0e-12;
        const double kx = g.xix[f];
        const double ky = g.xiy[f];
        const double kz = g.xiz[f];

        const double qc0[5] = {s.q.prs[c0], s.q.vx[c0], s.q.vy[c0], s.q.vz[c0], s.q.tmp[c0]};
        const double qc1[5] = {s.q.prs[c1], s.q.vx[c1], s.q.vy[c1], s.q.vz[c1], s.q.tmp[c1]};

        const double g0[5] = {
            s.grad.dprs_dx(c0) * kx + s.grad.dprs_dy(c0) * ky + s.grad.dprs_dz(c0) * kz,
            s.grad.dvx_dx(c0)  * kx + s.grad.dvx_dy(c0)  * ky + s.grad.dvx_dz(c0)  * kz,
            s.grad.dvy_dx(c0)  * kx + s.grad.dvy_dy(c0)  * ky + s.grad.dvy_dz(c0)  * kz,
            s.grad.dvz_dx(c0)  * kx + s.grad.dvz_dy(c0)  * ky + s.grad.dvz_dz(c0)  * kz,
            s.grad.dtmp_dx(c0) * kx + s.grad.dtmp_dy(c0) * ky + s.grad.dtmp_dz(c0) * kz
        };

        const double g1[5] = {
            s.grad.dprs_dx(c1) * kx + s.grad.dprs_dy(c1) * ky + s.grad.dprs_dz(c1) * kz,
            s.grad.dvx_dx(c1)  * kx + s.grad.dvx_dy(c1)  * ky + s.grad.dvx_dz(c1)  * kz,
            s.grad.dvy_dx(c1)  * kx + s.grad.dvy_dy(c1)  * ky + s.grad.dvy_dz(c1)  * kz,
            s.grad.dvz_dx(c1)  * kx + s.grad.dvz_dy(c1)  * ky + s.grad.dvz_dz(c1)  * kz,
            s.grad.dtmp_dx(c1) * kx + s.grad.dtmp_dy(c1) * ky + s.grad.dtmp_dz(c1) * kz
        };

        for (std::size_t v = 0; v < kNumPrimitives; ++v) {
            const double LL = qc1[v] - 2.0 * g0[v];
            const double RR = qc0[v] + 2.0 * g1[v];

            // Reconstruct qL
            const double dminus_L = qc0[v] - LL;
            const double dplus_L  = qc1[v] - qc0[v];
            if (std::abs(dminus_L) > kEps && std::abs(dplus_L) > kEps) {
                const double rL = dplus_L / dminus_L;
                qL[v] = qc0[v] + 0.5 * Limiter1D::phi(rL) * dminus_L;
            } else {
                qL[v] = qc0[v];
            }

            // Reconstruct qR
            const double dminus_R = qc1[v] - RR;
            const double dplus_R  = qc0[v] - qc1[v];
            if (std::abs(dminus_R) > kEps && std::abs(dplus_R) > kEps) {
                const double rR = dplus_R / dminus_R;
                qR[v] = qc1[v] + 0.5 * Limiter1D::phi(rR) * dminus_R;
            } else {
                qR[v] = qc1[v];
            }
        }
    }

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

static_assert(ReconstructionPolicy<MusclDirectional<limiter::Minmod1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::VanLeer1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::Superbee1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::VanAlbada1D>>);

} // namespace cfd::solver::recon
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/fields/fields_view.hpp"
#include "cfd/solver/limiter/limiters.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver::recon {

// --- Directional MUSCL Reconstruction Policy ---------------------------------

template <typename Limiter1D>
struct MusclDirectional {
    static constexpr bool kNeedsGradients = true;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::FaceCellDistanceVector;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "MUSCL_DIRECTIONAL"; }
    static constexpr const char* limiter_name() noexcept { return Limiter1D::name(); }

    // No cell-wise limiter sweep needed -> zero cost!
    static inline void compute_limiters(const mesh::MeshPart& /*mesh*/,
                                 const mesh::MeshAuxConnectivity& /*aux_conn*/,
                                 fields::ConstPrimitiveView /*q*/,
                                 fields::ConstPrimitiveGradView /*grad*/,
                                 fields::PrimitiveView<double> /*phi*/,
                                 const double /*venkat_k = 1.0*/) noexcept {}

    static inline void face_states(const ReconField& s,
                            const mesh::MeshPart& /*mesh*/,
                            const mesh::MeshAuxGeometry& aux_geom,
                            const std::size_t f,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[constants::kNumVars],
                            double qR[constants::kNumVars]) noexcept {
        //constexpr double kEps = 1.0e-12;
        const double kx = aux_geom.face_cell_dist_x[f];
        const double ky = aux_geom.face_cell_dist_y[f];
        const double kz = aux_geom.face_cell_dist_z[f];

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

        for (std::size_t v = 0; v < constants::kNumVars; ++v) {
            const double LL = qc1[v] - 2.0 * g0[v];
            const double RR = qc0[v] + 2.0 * g1[v];

            qL[v] = muscl2(LL, qc0[v], qc1[v]);
            qR[v] = muscl2(RR, qc1[v], qc0[v]);
        }
    }

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


    private:
    static inline double muscl2(double a_mm, double a_m, double a) {
        constexpr double eps = double(1e-10);
        const double dminus = a_m - a_mm;
        const double dplus  = a - a_m;
        if (std::fabs(dminus) > eps && std::fabs(dplus) > eps) {
            const double r = dplus/dminus;
            return a_m + double(0.5)*Limiter1D::phi(r)*dminus;
        }
        return a_m;
    }
};

static_assert(ReconstructionPolicy<MusclDirectional<limiter::Minmod1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::VanLeer1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::Superbee1D>>);
static_assert(ReconstructionPolicy<MusclDirectional<limiter::VanAlbada1D>>);

} // namespace cfd::solver::recon
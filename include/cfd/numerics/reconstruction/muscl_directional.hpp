// =============================================================================
// Directional MUSCL Reconstruction Policy (Face-normal stencil)
// =============================================================================
#pragma once

#include <cmath>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/numerics/limiter/tvd_1d_limiters.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"

namespace cfd::numerics::recon {

template <typename Limiter1D>
struct MusclDirectional {
    static constexpr bool kNeedsGradients = true;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::FaceCellDistanceVector;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "MUSCL_DIRECTIONAL"; }
    static constexpr const char* limiter_name() noexcept { return Limiter1D::name(); }

    template <std::size_t NVars>
    static inline void compute_limiters(
        const mesh::MeshPart& /*mesh*/,
        const mesh::MeshAuxConnectivity& /*aux_conn*/,
        const double* CFD_RESTRICT const* CFD_RESTRICT /*q*/,
        const double* CFD_RESTRICT const* CFD_RESTRICT /*grad_x*/,
        const double* CFD_RESTRICT const* CFD_RESTRICT /*grad_y*/,
        const double* CFD_RESTRICT const* CFD_RESTRICT /*grad_z*/,
        double* CFD_RESTRICT const* CFD_RESTRICT /*phi*/,
        const double /*venkat_k = 1.0*/) noexcept {}

    template <std::size_t NVars>
    static inline void face_states(
        const ReconBatchField<NVars>& s,
        const mesh::MeshPart& /*mesh*/,
        const mesh::MeshAuxGeometry& aux_geom,
        const std::size_t f,
        const std::size_t c0,
        const std::size_t c1,
        double* CFD_RESTRICT qL,
        double* CFD_RESTRICT qR) noexcept {
        const double kx = aux_geom.face_cell_dist_x[f];
        const double ky = aux_geom.face_cell_dist_y[f];
        const double kz = aux_geom.face_cell_dist_z[f];

        for (std::size_t v = 0; v < NVars; ++v) {
            const double val0 = s.q[v][c0];
            const double val1 = s.q[v][c1];

            const double g0 = s.grad_x[v][c0] * kx + 
                              s.grad_y[v][c0] * ky + 
                              s.grad_z[v][c0] * kz;
            const double g1 = s.grad_x[v][c1] * kx + 
                              s.grad_y[v][c1] * ky + 
                              s.grad_z[v][c1] * kz;

            const double LL = val1 - 2.0 * g0;
            const double RR = val0 + 2.0 * g1;

            qL[v] = muscl2(LL, val0, val1);
            qR[v] = muscl2(RR, val1, val0);
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

private:
    static inline double muscl2(const double a_mm, const double a_m, const double a) noexcept {
        constexpr double eps = 1.0e-10;
        const double dminus = a_m - a_mm;
        const double dplus  = a - a_m;
        if (std::fabs(dminus) > eps && std::fabs(dplus) > eps) {
            const double r = dplus / dminus;
            return a_m + 0.5 * Limiter1D::phi(r) * dminus;
        }
        return a_m;
    }
};

static_assert(ReconstructionPolicy<MusclDirectional<numerics::limiter::tvd1d::Minmod>, 1>);
static_assert(ReconstructionPolicy<MusclDirectional<numerics::limiter::tvd1d::Minmod>, 5>);
static_assert(ReconstructionPolicy<MusclDirectional<numerics::limiter::tvd1d::VanLeer>, 5>);
static_assert(ReconstructionPolicy<MusclDirectional<numerics::limiter::tvd1d::Superbee>, 5>);
static_assert(ReconstructionPolicy<MusclDirectional<numerics::limiter::tvd1d::VanAlbada>, 5>);

} // namespace cfd::numerics::recon
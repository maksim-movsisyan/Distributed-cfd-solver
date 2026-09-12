#pragma once

#include <cstddef>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"

namespace cfd::numerics::recon {

struct FirstOrder {
    static constexpr bool kNeedsGradients = false;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "FIRST_ORDER"; }
    static constexpr const char* limiter_name() noexcept { return "NONE"; }

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
        const mesh::MeshAuxGeometry& /*aux_geom*/,
        const std::size_t /*f*/,
        const std::size_t c0,
        const std::size_t c1,
        double* CFD_RESTRICT qL,
        double* CFD_RESTRICT qR) noexcept {
        for (std::size_t v = 0; v < NVars; ++v) {
            qL[v] = s.q[v][c0];
            qR[v] = s.q[v][c1];
        }
    }

    template <std::size_t NVars>
    static inline void boundary_face_states(
        const ReconBatchField<NVars>& s,
        const mesh::MeshPart& mesh,
        const mesh::MeshAuxGeometry& aux_geom,
        const std::size_t f,
        const std::size_t c0,
        const std::size_t cg,
        double* CFD_RESTRICT qL,
        double* CFD_RESTRICT qR) noexcept {
        face_states<NVars>(s, mesh, aux_geom, f, c0, cg, qL, qR);
    }
};

static_assert(ReconstructionPolicy<FirstOrder, 1>);
static_assert(ReconstructionPolicy<FirstOrder, 5>);

} // namespace cfd::numerics::recon
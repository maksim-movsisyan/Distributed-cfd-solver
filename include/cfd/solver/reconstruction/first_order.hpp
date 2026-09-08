// First-order (piecewise constant) reconstruction policy.
// The face states are the cell-centred values themselves: qL = q_c0, qR = q_c1.
// Zero arithmetic beyond the gather.
#pragma once

#include <cstddef>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver::recon {

/**
 * @struct FirstOrder
 * @brief Godunov-type 1st-order spatial reconstruction.
 */
struct FirstOrder {
    static constexpr bool kNeedsGradients = false;
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::None;
    static constexpr const char* name() noexcept { return "FIRST_ORDER"; }
    static constexpr const char* limiter_name() noexcept { return "NONE"; }

    static inline void compute_limiters(const mesh::MeshPart& /*mesh*/,
                                    const mesh::MeshAuxConnectivity& /*aux_conn*/,
                                    fields::ConstPrimitiveView /*q*/,
                                    fields::ConstPrimitiveGradView /*grad*/,
                                    fields::PrimitiveView<double> /*phi*/,
                                    const double /*venkat_k = 1.0*/) noexcept {}
                                    
    /**
     * @brief Gather primitive states on interior faces [0, n_inner_faces).
     */
    static inline void face_states(const ReconField& s,
                            const mesh::MeshPart& /*mesh*/,
                            const mesh::MeshAuxGeometry& /*aux_geom*/,
                            const std::size_t /*f*/,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[constants::kNumVars],
                            double qR[constants::kNumVars]) noexcept {
        qL[0] = s.q.prs[c0];
        qL[1] = s.q.vx[c0];
        qL[2] = s.q.vy[c0];
        qL[3] = s.q.vz[c0];
        qL[4] = s.q.tmp[c0];

        qR[0] = s.q.prs[c1];
        qR[1] = s.q.vx[c1];
        qR[2] = s.q.vy[c1];
        qR[3] = s.q.vz[c1];
        qR[4] = s.q.tmp[c1];
    }

    /**
     * @brief Gather primitive states on boundary faces [n_inner_faces, n_faces).
     */
    static inline void boundary_face_states(const ReconField& s,
                                     const mesh::MeshPart& mesh,
                                     const mesh::MeshAuxGeometry& aux_geom,
                                     const std::size_t f,
                                     const std::size_t c0,
                                     const std::size_t cg,
                                     double qL[constants::kNumVars],
                                     double qR[constants::kNumVars]) noexcept {
        face_states(s, mesh, aux_geom, f, c0, cg, qL, qR);
    }
};

static_assert(ReconstructionPolicy<FirstOrder>);

} // namespace cfd::solver::recon
// First-order (piecewise constant) reconstruction policy.
// The face states are the cell-centred values themselves: qL = q_c0, qR = q_c1.
// Zero arithmetic beyond the gather.
#pragma once

#include <cstddef>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"

namespace cfd::solver::recon {

/**
 * @struct FirstOrder
 * @brief Godunov-type 1st-order spatial reconstruction.
 */
struct FirstOrder {
    static constexpr bool kNeedsGradients = false;

    static constexpr const char* name() noexcept { return "FIRST_ORDER"; }
    static constexpr const char* limiter_name() noexcept { return "NONE"; }

    /**
     * @brief Empty geometry placeholder (0 bytes in execution, fully elided by compiler).
     */
    struct Geometry {
        Geometry() = default;
        explicit Geometry(const mesh::MeshPart& /*mp*/) noexcept {}
    };

    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& mp,
                                                 const double /*venkat_k*/ = 0.0) noexcept {
        return Geometry(mp);
    }

    /**
     * @brief Gather primitive states on interior faces [0, n_inner_faces).
     */
    static void face_states(const ReconField& s,
                            const Geometry& /*g*/,
                            const std::size_t /*f*/,
                            const std::size_t c0,
                            const std::size_t c1,
                            double qL[kNumPrimitives],
                            double qR[kNumPrimitives]) noexcept {
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
    static void boundary_face_states(const ReconField& s,
                                     const Geometry& g,
                                     const std::size_t f,
                                     const std::size_t c0,
                                     const std::size_t cg,
                                     double qL[kNumPrimitives],
                                     double qR[kNumPrimitives]) noexcept {
        face_states(s, g, f, c0, cg, qL, qR);
    }
};

static_assert(ReconstructionPolicy<FirstOrder>);

} // namespace cfd::solver::recon
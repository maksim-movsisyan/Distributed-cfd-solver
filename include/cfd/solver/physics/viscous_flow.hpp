#pragma once

#include <cmath>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"

namespace cfd::solver::physics {

/**
 * @struct ViscousFlow
 * @brief Navier-Stokes equations: pure viscous mean flow.
 */
struct ViscousFlow {
    double prandtl = constants::kAirPrandtl; ///< Molecular Prandtl number [-]

    static constexpr std::size_t kNumVars = static_cast<std::size_t>(constants::kNumVars);
    static constexpr bool kHasViscous = true;
    static constexpr bool kNeedsGradients = true;
    static constexpr const char* name() noexcept { return "VISCOUS_FLOW"; }

    static constexpr std::size_t kNumExtraVars = 0;
    static constexpr bool kHasEddyViscosity = false;
    static constexpr bool kNeedsFaceMdot = false;
    static constexpr bool kNeedsWallDist = false;

    /**
     * @struct Geometry
     * @brief Mesh-fixed data for face-gradient correction and the viscous
     *        time-step estimate.
     */
    struct Geometry {
        std::vector<double> dx, dy, dz; ///< [n_faces] centroid displacement x_R - x_L
        std::vector<double> inv_d;     ///< [n_faces] 1 / |d| (correction weight)
    };

    /**
     * @brief Precomputes face displacement vectors once at startup.
     *
     * Interior faces use owner->neighbour centroid offsets (neighbours may be
     * halo ghosts, which carry centroids). Boundary faces mirror through the
     * face centroid: d = 2 (x_face - x_owner).
     */
    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& mp) {
        const std::size_t n_inner = static_cast<std::size_t>(mp.n_inner_faces);
        const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);

        Geometry g;
        g.dx.resize(n_faces);
        g.dy.resize(n_faces);
        g.dz.resize(n_faces);
        g.inv_d.resize(n_faces);

        for (std::size_t f = 0; f < n_faces; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);

            double ex, ey, ez;
            if (f < n_inner) {
                const std::size_t c1 = static_cast<std::size_t>(mp.face_neigh[f]);
                ex = mp.cell_centroid_x[c1] - mp.cell_centroid_x[c0];
                ey = mp.cell_centroid_y[c1] - mp.cell_centroid_y[c0];
                ez = mp.cell_centroid_z[c1] - mp.cell_centroid_z[c0];
            } else {
                // Mirrored virtual ghost: twice the owner->face offset
                ex = 2.0 * (mp.face_centroid_x[f] - mp.cell_centroid_x[c0]);
                ey = 2.0 * (mp.face_centroid_y[f] - mp.cell_centroid_y[c0]);
                ez = 2.0 * (mp.face_centroid_z[f] - mp.cell_centroid_z[c0]);
            }

            const double d2 = std::max(ex * ex + ey * ey + ez * ez, 1.0e-30);
            g.dx[f] = ex;
            g.dy[f] = ey;
            g.dz[f] = ez;
            g.inv_d[f] = 1.0 / std::sqrt(d2);
        }
        return g;
    }

    /** @brief Sutherland dynamic viscosity [Pa s]. */
    [[nodiscard]] static double viscosity(const double T) noexcept {
        constexpr double kSuthConst = constants::kSutherlandViscosity 
                                    * (constants::kSutherlandTRef + constants::kSutherlandT);
        const double ratio = T / constants::kSutherlandTRef;
        return kSuthConst * ratio * std::sqrt(ratio) / (T + constants::kSutherlandT);
    }

    /** @brief Constant Prandtl thermal conductivity [W / (m K)]. */
    template <eos::EquationOfState EOS>
    [[nodiscard]] static double thermal_conductivity(const EOS& eos, 
                                                    const double T, const double p, 
                                                    const double prandtl) noexcept {
        return (viscosity(T) * eos.cp_Tp(T, p)) / prandtl;
    }

    template <eos::EquationOfState EOS>
    [[nodiscard]] double thermal_conductivity(const EOS& eos, const double T, const double p) const noexcept {
        return (viscosity(T) * eos.cp_Tp(T, p)) / prandtl;
    }
};

static_assert(PhysicsGeneral<ViscousFlow>);

} // namespace cfd::solver::physics
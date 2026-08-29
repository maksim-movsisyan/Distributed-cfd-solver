// Gradient limiter POLICIES for MUSCL 2nd-order spatial reconstruction.
//
// Contract (pure scalar math, inlined into the face/cell limiter sweep):
//   phi(delta_nb, delta_face, eps2) -> gradient limiter scale in [0, ~1.5]
//
//   delta_nb   = (q_max - q_i) if delta_face > 0, else (q_min - q_i):
//                admissible increment towards the neighbor extremum.
//   delta_face = unlimited reconstructed face increment: grad(q) . r_f
//   eps2       = smoothing parameter (K * h)^3 to preserve convergence.
#pragma once

namespace cfd::solver::limiter {

/**
 * @brief Barth-Jespersen (1989): Strict TVD limiter, min(1, r).
 * Non-smooth, strictly monotone, most dissipative near extrema.
 */
struct BarthJespersen {
    static constexpr const char* name() noexcept { return "BARTH_JESPERSEN"; }

    [[nodiscard]] static constexpr double phi(const double delta_nb,
                                              const double delta_face,
                                              const double /*eps2*/ = 0.0) noexcept {
        if (delta_face == 0.0) {
            return 1.0;
        }

        const double r = delta_nb / delta_face;
        if (r <= 0.0) {
            return 0.0;
        }
        return r < 1.0 ? r : 1.0;
    }
};

/**
 * @brief Venkatakrishnan (1993): Smooth limiter preventing stall of residuals.
 * phi -> 1 in smooth regions, asymptotically bounds increments by delta_nb near shocks.
 */
struct Venkatakrishnan {
    static constexpr const char* name() noexcept { return "VENKATAKRISHNAN"; }

    [[nodiscard]] static constexpr double phi(const double delta_nb,
                                              const double delta_face,
                                              const double eps2) noexcept {
        if (delta_face == 0.0) {
            return 1.0;
        }

        const double prod = delta_nb * delta_face;
        if (prod <= 0.0) {
            return 0.0;
        }

        const double a2 = delta_nb * delta_nb + eps2;
        const double num = a2 + 2.0 * prod;
        const double den = a2 + 2.0 * delta_face * delta_face + prod;

        return num / den;
    }
};

/**
 * @brief Van Albada: Smooth quadratic limiter adapted for unstructured meshes.
 */
struct VanAlbada {
    static constexpr const char* name() noexcept { return "VAN_ALBADA"; }

    [[nodiscard]] static constexpr double phi(const double delta_nb,
                                              const double delta_face,
                                              const double eps2 = 0.0) noexcept {
        if (delta_face == 0.0) {
            return 1.0;
        }

        const double prod = delta_nb * delta_face;
        if (prod <= 0.0) {
            return 0.0;
        }

        const double den = delta_nb * delta_nb + delta_face * delta_face + eps2;
        if (den == 0.0) {
            return 1.0;
        }

        const double num = delta_nb * delta_nb + prod;
        return num / den;
    }
};

} // namespace cfd::solver::limiter
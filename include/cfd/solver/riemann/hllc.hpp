// HLLC approximate Riemann solver (E.F. Toro) with Roe-averaged outer wave speeds.
//
// This is a zero-overhead flux policy struct designed to be injected into
// ResidualKernel at compile time. All methods inline directly into face sweeps.
#pragma once

#include <algorithm>
#include <cmath>

#include "cfd/core/types.hpp"
#include "cfd/solver/eos/state_conversions.hpp"

namespace cfd::solver::riemann {

struct HllcFlux {
    [[nodiscard]] static constexpr const char* name() noexcept { return "HLLC"; }

    // Computes numerical convective flux F = area * F_num(UL, UR, n)
    // and returns max wave speed estimate smax for local CFL time stepping.
    template <eos::EquationOfState EOS>
    static inline void face_flux(const EOS& eos,
                                 const double UL[constants::kNumVars],
                                 const double UR[constants::kNumVars],
                                 const double nx,
                                 const double ny,
                                 const double nz,
                                 const double area,
                                 double F[constants::kNumVars],
                                 double& smax) noexcept {
        // --- 1. Primitive reconstruction & Fast Inverses ---
        const double rhoL = UL[0];
        const double rhoR = UR[0];
        const double inv_rhoL = 1.0 / rhoL;
        const double inv_rhoR = 1.0 / rhoR;

        const double pL = eos::pressure(eos, UL);
        const double pR = eos::pressure(eos, UR);

        const double uLx = UL[1] * inv_rhoL;
        const double uLy = UL[2] * inv_rhoL;
        const double uLz = UL[3] * inv_rhoL;

        const double uRx = UR[1] * inv_rhoR;
        const double uRy = UR[2] * inv_rhoR;
        const double uRz = UR[3] * inv_rhoR;

        const double unL = uLx * nx + uLy * ny + uLz * nz;
        const double unR = uRx * nx + uRy * ny + uRz * nz;

        const double aL = eos.sound_speed_rhop(rhoL, pL);
        const double aR = eos.sound_speed_rhop(rhoR, pR);

        // --- 2. Roe-averaged Outer Wave Estimates ---
        const double rl = std::sqrt(rhoL);
        const double rr = std::sqrt(rhoR);
        const double w  = 1.0 / (rl + rr);

        const double roe_ux = (rl * uLx + rr * uRx) * w;
        const double roe_uy = (rl * uLy + rr * uRy) * w;
        const double roe_uz = (rl * uLz + rr * uRz) * w;

        const double HL = (UL[4] + pL) * inv_rhoL;
        const double HR = (UR[4] + pR) * inv_rhoR;
        const double roe_h = (rl * HL + rr * HR) * w;

        const double q2 = roe_ux * roe_ux + roe_uy * roe_uy + roe_uz * roe_uz;
        const double roe_a  = eos.sound_speed_Hv2(roe_h, q2);
        const double roe_un = roe_ux * nx + roe_uy * ny + roe_uz * nz;

        // Wave speed bounds (Einfeldt / Davis)
        const double SL = std::min(unL - aL, roe_un - roe_a);
        const double SR = std::max(unR + aR, roe_un + roe_a);
        smax = std::max(std::fabs(SL), std::fabs(SR));

        // --- 3. Star (Contact) Wave Speed S_M ---
        const double dL = SL - unL;
        const double dR = SR - unR;
        const double denom = rhoL * dL - rhoR * dR;

        // Safe division guard against vacuum/degenerate states
        const double SM = (std::fabs(denom) > 1e-14)
            ? (pR - pL + rhoL * unL * dL - rhoR * unR * dR) / denom
            : 0.5 * (unL + unR);

        // --- 4. Branching & Lazy Flux Evaluation ---
        if (SM >= 0.0) {
            // Left state or Left Star state
            const double FL[constants::kNumVars] = {
                rhoL * unL,
                rhoL * uLx * unL + pL * nx,
                rhoL * uLy * unL + pL * ny,
                rhoL * uLz * unL + pL * nz,
                (UL[4] + pL) * unL
            };

            if (SL >= 0.0) {
                // Supersonic Left -> Right
                for (int v = 0; v < constants::kNumVars; ++v) {
                    F[v] = FL[v] * area;
                }
            } else {
                // Subsonic Left Star (F*_L)
                star_flux(UL, FL, rhoL, pL, uLx, uLy, uLz, unL, aL, SL, SM, nx, ny, nz, area, F);
            }
        } else {
            // Right state or Right Star state
            const double FR[constants::kNumVars] = {
                rhoR * unR,
                rhoR * uRx * unR + pR * nx,
                rhoR * uRy * unR + pR * ny,
                rhoR * uRz * unR + pR * nz,
                (UR[4] + pR) * unR
            };

            if (SR <= 0.0) {
                // Supersonic Right -> Left
                for (int v = 0; v < constants::kNumVars; ++v) {
                    F[v] = FR[v] * area;
                }
            } else {
                // Subsonic Right Star (F*_R)
                star_flux(UR, FR, rhoR, pR, uRx, uRy, uRz, unR, aR, SR, SM, nx, ny, nz, area, F);
            }
        }
    }

    /**
     * @brief Smooth approximate flux Jacobians dF/dUL and dF/dUR (5x5,
     *        row-major) for implicit schemes — Rusanov-type linearization
     *        with frozen wave speed:
     *
     *   F ≈ 0.5 (Fp(UL) + Fp(UR)) - 0.5 smax (UR - UL)
     *   dF/dUL = A [ 0.5 Jp(UL) - 0.5 smax I ]
     *   dF/dUR = A [ 0.5 Jp(UR) + 0.5 smax I ]
     *
     * where Fp is the physical normal flux and Jp its (smooth!) Jacobian,
     * obtained by central differences of Fp. Finite-differencing the REAL
     * HLLC flux instead is branch-consistent but kink-sensitive: near sonic /
     * star-state branch boundaries a 1e-6 step produces spurious derivatives
     * orders of magnitude above the physical scale, which wreck the implicit
     * preconditioner. The Rusanov linearization is the standard robust
     * implicit-operator approximation; the residual itself keeps using the
     * exact HLLC flux. smax comes from one face_flux call (Einfeldt bounds).
     */
    template <eos::EquationOfState EOS>
    static inline void face_flux_jacobian(const EOS& eos,
                                          const double UL[constants::kNumVars],
                                          const double UR[constants::kNumVars],
                                          const double nx,
                                          const double ny,
                                          const double nz,
                                          const double area,
                                          double dFL[constants::kNumVars * constants::kNumVars],
                                          double dFR[constants::kNumVars * constants::kNumVars]) noexcept {
        constexpr int N = constants::kNumVars;

        // Frozen outer wave speed from the flux policy itself.
        double F0[N];
        double smax = 0.0;
        face_flux(eos, UL, UR, nx, ny, nz, area, F0, smax);

        physical_flux_jacobian(eos, UL, nx, ny, nz, dFL);
        physical_flux_jacobian(eos, UR, nx, ny, nz, dFR);

        const double cL = 0.5 * area;
        const double cR = 0.5 * area;
        for (int i = 0; i < N * N; ++i) {
            dFL[i] *= cL;
            dFR[i] *= cR;
        }
        for (int i = 0; i < N; ++i) {
            dFL[i * N + i] -= 0.5 * smax * area;
            dFR[i * N + i] += 0.5 * smax * area;
        }
    }

    /**
     * @brief Smooth Rusanov (local Lax–Friedrichs) face flux — branch-free
     *        surrogate used where implicit linearizations need a differentiable
     *        flux (e.g. boundary blocks differentiated through the BC ghost).
     */
    template <eos::EquationOfState EOS>
    static inline void rusanov_face_flux(const EOS& eos,
                                         const double UL[constants::kNumVars],
                                         const double UR[constants::kNumVars],
                                         const double nx, const double ny, const double nz,
                                         const double area,
                                         double F[constants::kNumVars]) noexcept {
        constexpr int N = constants::kNumVars;

        const double rhoL = UL[0];
        const double rhoR = UR[0];
        const double uLn = UL[1] * nx + UL[2] * ny + UL[3] * nz;
        const double uRn = UR[1] * nx + UR[2] * ny + UR[3] * nz;
        const double aL = eos.sound_speed_rhop(rhoL, eos::pressure(eos, UL));
        const double aR = eos.sound_speed_rhop(rhoR, eos::pressure(eos, UR));
        const double s = std::max(std::fabs(uLn) + aL, std::fabs(uRn) + aR);

        double FL[N], FR[N];
        physical_normal_flux(eos, UL, nx, ny, nz, FL);
        physical_normal_flux(eos, UR, nx, ny, nz, FR);

        for (int v = 0; v < N; ++v) {
            F[v] = area * (0.5 * (FL[v] + FR[v]) - 0.5 * s * (UR[v] - UL[v]));
        }
    }

private:
    /** @brief Central-difference Jacobian of the smooth physical normal flux. */
    template <eos::EquationOfState EOS>
    static inline void physical_flux_jacobian(const EOS& eos,
                                              const double U[constants::kNumVars],
                                              const double nx, const double ny,
                                              const double nz,
                                              double J[constants::kNumVars * constants::kNumVars]) noexcept {
        constexpr int N = constants::kNumVars;
        constexpr double kStep = 1.0e-6;

        double Up[N];
        for (int v = 0; v < N; ++v) {
            Up[v] = U[v];
        }

        for (int k = 0; k < N; ++k) {
            const double base = Up[k];
            const double h = kStep * (std::fabs(base) + 1.0);

            Up[k] = base + h;
            double Fp[N];
            physical_normal_flux(eos, Up, nx, ny, nz, Fp);

            Up[k] = base - h;
            double Fm[N];
            physical_normal_flux(eos, Up, nx, ny, nz, Fm);

            Up[k] = base;
            const double inv_2h = 1.0 / (2.0 * h);
            for (int i = 0; i < N; ++i) {
                J[i * N + k] = (Fp[i] - Fm[i]) * inv_2h;
            }
        }
    }

    /** @brief Physical (smooth, branch-free) convective normal flux. */
    template <eos::EquationOfState EOS>
    static inline void physical_normal_flux(const EOS& eos,
                                            const double U[constants::kNumVars],
                                            const double nx, const double ny, const double nz,
                                            double F[constants::kNumVars]) noexcept {
        const double rho = U[0];
        const double inv_rho = 1.0 / rho;
        const double ux = U[1] * inv_rho;
        const double uy = U[2] * inv_rho;
        const double uz = U[3] * inv_rho;
        const double un = ux * nx + uy * ny + uz * nz;
        const double p = eos::pressure(eos, U);

        F[0] = rho * un;
        F[1] = rho * ux * un + p * nx;
        F[2] = rho * uy * un + p * ny;
        F[3] = rho * uz * un + p * nz;
        F[4] = (U[4] + p) * un;
    }

    static inline void star_flux(const double UK[constants::kNumVars],
                                 const double FK[constants::kNumVars],
                                 const double rhoK,
                                 const double pK,
                                 const double ux,
                                 const double uy,
                                 const double uz,
                                 const double unk,
                                 const double ak,
                                 const double SK,
                                 const double SM,
                                 const double nx,
                                 const double ny,
                                 const double nz,
                                 const double area,
                                 double F[constants::kNumVars]) noexcept {
        const double den = SK - SM;
        if (std::fabs(den) < 1.0e-6 * (ak + std::fabs(SK))) {
            for (int v = 0; v < constants::kNumVars; ++v) {
                F[v] = FK[v] * area;
            }
            return;
        }

        const double inv_den = 1.0 / den;
        const double d_wave  = SK - unk;
        const double q       = d_wave * inv_den;
        const double d       = SM - unk;

        const double p_term = pK / (rhoK * d_wave);
        const double e_star = UK[4] + rhoK * d * (SM + p_term);

        const double Ust[constants::kNumVars] = {
            q * rhoK,
            q * rhoK * (ux + d * nx),
            q * rhoK * (uy + d * ny),
            q * rhoK * (uz + d * nz),
            q * e_star
        };

        for (int v = 0; v < constants::kNumVars; ++v) {
            F[v] = (FK[v] + SK * (Ust[v] - UK[v])) * area;
        }
    }
};

} // namespace cfd::solver::riemann
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

private:
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
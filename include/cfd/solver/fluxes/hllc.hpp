// HLLC approximate Riemann solver (E.F. Toro) with Roe-averaged outer wave speeds
// and analytical Picard-Rusanov linearized Jacobians for implicit schemes.
#pragma once

#include <algorithm>
#include <cmath>

#include "cfd/solver/eos/concepts.hpp"
#include "cfd/solver/eos/state_conversions.hpp"

namespace cfd::solver::fluxes {

struct HllcFlux {
    [[nodiscard]] static constexpr const char* name() noexcept { return "HLLC"; }

    // =========================================================================
    // 1. Convective Residual Flux (Exact Non-linear HLLC)
    // =========================================================================
    // Computes numerical convective flux F = area * F_num(UL, UR, n)
    // and returns max wave speed estimate smax for local CFL time stepping.
    template <eos::EquationOfStatePolicy EOS>
    static inline void face_flux(const EOS& eos,
                                 const double UL[5],
                                 const double UR[5],
                                 const double nx, const double ny, const double nz,
                                 const double area,
                                 double F[5],
                                 double& smax) noexcept {
        // --- Primitive reconstruction & Fast Inverses ---
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

        // --- Roe-averaged Outer Wave Estimates ---
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

        // --- Star (Contact) Wave Speed S_M ---
        const double dL = SL - unL;
        const double dR = SR - unR;
        const double denom = rhoL * dL - rhoR * dR;

        // Safe division guard against vacuum/degenerate states
        const double SM = (std::fabs(denom) > 1e-14)
            ? (pR - pL + rhoL * unL * dL - rhoR * unR * dR) / denom
            : 0.5 * (unL + unR);

        // --- Branching & Lazy Flux Evaluation ---
        if (SM >= 0.0) {
            const double FL[5] = {
                rhoL * unL,
                rhoL * uLx * unL + pL * nx,
                rhoL * uLy * unL + pL * ny,
                rhoL * uLz * unL + pL * nz,
                (UL[4] + pL) * unL
            };

            if (SL >= 0.0) {
                // Supersonic Left -> Right
                for (int v = 0; v < 5; ++v) {
                    F[v] = FL[v] * area;
                }
            } else {
                // Subsonic Left Star (F*_L)
                star_flux(UL, FL, rhoL, pL, uLx, uLy, uLz, unL, aL, SL, SM, nx, ny, nz, area, F);
            }
        } else {
            const double FR[5] = {
                rhoR * unR,
                rhoR * uRx * unR + pR * nx,
                rhoR * uRy * unR + pR * ny,
                rhoR * uRz * unR + pR * nz,
                (UR[4] + pR) * unR
            };

            if (SR <= 0.0) {
                // Supersonic Right -> Left
                for (int v = 0; v < 5; ++v) {
                    F[v] = FR[v] * area;
                }
            } else {
                // Subsonic Right Star (F*_R)
                star_flux(UR, FR, rhoR, pR, uRx, uRy, uRz, unR, aR, SR, SM, nx, ny, nz, area, F);
            }
        }
    }

    // =========================================================================
    // 2. Analytical Implicit Operator (LHS Matrix Assembly)
    // =========================================================================
    // Computes smooth approximate Jacobians dF/dUL and dF/dUR (5x5, row-major)
    // for implicit matrix systems:
    //
    //   dF/dUL = Area * [ 0.5 * Jp(UL) + 0.5 * smax * I ]
    //   dF/dUR = Area * [ 0.5 * Jp(UR) - 0.5 * smax * I ]
    //
    // Guaranteed M-matrix property on the diagonal, non-zero coupling across
    // the face, and zero finite-difference overhead.
    template <eos::EquationOfStatePolicy EOS>
    static inline void face_flux_jacobian(const EOS& eos,
                                          const double UL[5],
                                          const double UR[5],
                                          const double nx, const double ny, const double nz,
                                          const double area,
                                          double dFL[25],
                                          double dFR[25]) noexcept {
        constexpr int N = 5;

        // Frozen wave speed estimation from current states
        double F0[N];
        double smax = 0.0;
        face_flux(eos, UL, UR, nx, ny, nz, area, F0, smax);

        // Analytical physical Euler Jacobians: Jp = d(F_p · n)/dU
        analytical_physical_jacobian(eos, UL, nx, ny, nz, dFL);
        analytical_physical_jacobian(eos, UR, nx, ny, nz, dFR);

        const double half_area = 0.5 * area;
        const double diss = 0.5 * smax * area;

        for (int i = 0; i < N * N; ++i) {
            dFL[i] *= half_area;
            dFR[i] *= half_area;
        }

        // Add stabilizing diagonal dissipation:
        // +diss for owner (diagonal boost), -diss for neighbor
        for (int i = 0; i < N; ++i) {
            dFL[i * N + i] += diss;
            dFR[i * N + i] -= diss;
        }
    }

private:
    // Closed-form analytical Jacobian of normal physical Euler flux (5x5, row-major)
    template <eos::EquationOfStatePolicy EOS>
    static inline void analytical_physical_jacobian(const EOS& eos,
                                                    const double U[5],
                                                    const double nx,
                                                    const double ny,
                                                    const double nz,
                                                    double J[25]) noexcept {
        const double rho = U[0];
        const double inv_rho = 1.0 / rho;
        const double u = U[1] * inv_rho;
        const double v = U[2] * inv_rho;
        const double w = U[3] * inv_rho;
        const double E = U[4] * inv_rho;

        const double p = eos::pressure(eos, U);
        const double gamma = eos.gamma();
        const double phi = gamma - 1.0;

        const double un = u * nx + v * ny + w * nz;
        const double q2 = u * u + v * v + w * w;
        const double H  = E + p * inv_rho;
        const double half_phi_q2 = 0.5 * phi * q2;

        // Row 0: Continuity (rho * un)
        J[0]  = 0.0;
        J[1]  = nx;
        J[2]  = ny;
        J[3]  = nz;
        J[4]  = 0.0;

        // Row 1: X-Momentum (rho * u * un + p * nx)
        J[5]  = half_phi_q2 * nx - u * un;
        J[6]  = un + nx * (1.0 - phi) * u;
        J[7]  = u * ny - phi * nx * v;
        J[8]  = u * nz - phi * nx * w;
        J[9]  = phi * nx;

        // Row 2: Y-Momentum (rho * v * un + p * ny)
        J[10] = half_phi_q2 * ny - v * un;
        J[11] = v * nx - phi * ny * u;
        J[12] = un + ny * (1.0 - phi) * v;
        J[13] = v * nz - phi * ny * w;
        J[14] = phi * ny;

        // Row 3: Z-Momentum (rho * w * un + p * nz)
        J[15] = half_phi_q2 * nz - w * un;
        J[16] = w * nx - phi * nz * u;
        J[17] = w * ny - phi * nz * v;
        J[18] = un + nz * (1.0 - phi) * w;
        J[19] = phi * nz;

        // Row 4: Energy ((rho * E + p) * un)
        J[20] = un * (half_phi_q2 - H);
        J[21] = H * nx - phi * u * un;
        J[22] = H * ny - phi * v * un;
        J[23] = H * nz - phi * w * un;
        J[24] = gamma * un;
    }

    static inline void star_flux(const double UK[5],
                                 const double FK[5],
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
                                 double F[5]) noexcept {
        const double den = SK - SM;
        if (std::fabs(den) < 1.0e-6 * (ak + std::fabs(SK))) {
            for (int v = 0; v < 5; ++v) {
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

        const double Ust[5] = {
            q * rhoK,
            q * rhoK * (ux + d * nx),
            q * rhoK * (uy + d * ny),
            q * rhoK * (uz + d * nz),
            q * e_star
        };

        for (int v = 0; v < 5; ++v) {
            F[v] = (FK[v] + SK * (Ust[v] - UK[v])) * area;
        }
    }
};

} // namespace cfd::solver::fluxes
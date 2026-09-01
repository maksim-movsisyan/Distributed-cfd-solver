// Spalart-Allmaras one-equation turbulence model (SA, 1994, no trip terms).
//
// A self-contained PHYSICS MODULE (design-01): owns its transported variable
// rho*nu_tilde (conservative update blocks), its boundary conditions, its
// gradient plane, its wall distance, and its kernels. Coupling:
//   - consumes the mean-flow FACE MASS FLUX (first-order upwind convection —
//     the same discrete convection operator as the mean flow, no second
//     Riemann solve);
//   - consumes mean-flow velocity gradients (production term);
//   - provides the eddy viscosity mut[c] consumed by the viscous flux.
//
// Working-variable chain (NASA-standard formulation, S = strain magnitude):
//   chi = nu_tilde / nu,   fv1 = chi^3 / (chi^3 + cv1^3)
//   fv2 = 1 - chi / (1 + chi * fv1),   Sbar = S + nu_tilde * fv2 / (kappa^2 d^2)
//   r~  = nu_tilde / (Sbar * kappa^2 * d^2)  clipped to [0, 10]
//   g   = r~ + cw2 (r~^6 - r~),   fw = g * ((1 + cw3^6)/(g^6 + cw3^6))^(1/6)
//   P   = cb1 * Sbar * (rho nu_tilde)                 (production)
//   D   = cw1 * fw * (rho nu_tilde) * nu_tilde / d^2  (destruction)
//   nu_t = nu_tilde * fv1
//
// Stability: post-stage clamp rho*nu_tilde >= 0 and Sbar floored — the
// conservative "positive SA" practice.
#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/fields/fields_manager.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/halo.hpp"
#include "cfd/solver/physics/viscous_flow.hpp"
#include "cfd/solver/turbulence/wall_distance.hpp"

namespace cfd::solver::turb {

/// Zone treatment of nu_tilde derived from the mean-flow boundary condition.
enum class NuZoneKind {
    Wall,      ///< No-slip wall: nu_ghost = -nu_interior (nu_wall = 0)
    Symmetry,  ///< Symmetry / slip wall: mirror
    Inflow,    ///< Freestream value nu_inf
    Outflow,   ///< Zero-gradient extrapolation
};

/**
 * @class SpalartAllmaras
 * @brief SA turbulence module: one transported variable, module-owned kernels.
 */
class SpalartAllmaras {
public:
    // --- Module config (parsed from [turbulence]) ---
    double nu_inf_ratio = 3.0;           ///< Freestream nu_tilde / nu_molecular [-]
    int max_distance_sweeps = 500;       ///< Wall-distance sweep budget
    double distance_tolerance = 1.0e-8;  ///< Wall-distance relative tolerance

    // --- Module metadata (satisfies physics::PhysicsGeneral like the stack
    //     base; a module carries no equation set of its own) ---
    static constexpr std::size_t kNumVars = 1;      ///< transported rho*nu_tilde
    static constexpr bool kHasViscous = false;      ///< not an equation set
    static constexpr std::size_t kNumExtraVars = 0; ///< carries no sub-modules
    static constexpr bool kNeedsGradients = true;   ///< its own nu_tilde plane
    static constexpr bool kNeedsFaceMdot = true;    ///< upwind convection
    static constexpr bool kNeedsWallDist = true;
    static constexpr bool kHasEddyViscosity = true; ///< provides mut
    static constexpr const char* name() noexcept { return "SA"; }

    struct Geometry {};
    [[nodiscard]] static Geometry build_geometry(const mesh::MeshPart& /*mp*/) noexcept {
        return {};
    }

    // --- Model constants (Spalart & Allmaras, 1994) ---
    static constexpr double kCb1 = 0.1355;
    static constexpr double kCb2 = 0.622;
    static constexpr double kSigma = 2.0 / 3.0;
    static constexpr double kKappa = 0.41;
    static constexpr double kCw2 = 0.3;
    static constexpr double kCw3 = 2.0;
    static constexpr double kCv1 = 7.1;
    static constexpr double kCw1 = kCb1 / (kKappa * kKappa) + (1.0 + kCb2) / kSigma;
    static constexpr double kCv1_3 = kCv1 * kCv1 * kCv1;

    // --- Registration (init time, deterministic order on all ranks) ---------

    void register_fields(fields::FieldsManager& mgr,
                         const std::size_t n_total,
                         const bool needs_prev_snapshot) {
        n_total_ = n_total;

        mgr.add_field<double>("rho_nu", n_total, fields::FieldLocation::Cell);
        mgr.add_field<double>("res_rho_nu", n_total, fields::FieldLocation::Cell);
        if (needs_prev_snapshot) {
            mgr.add_field<double>("prev_rho_nu", n_total, fields::FieldLocation::Cell);
        }
        mgr.add_field<double>("stage_rho_nu", n_total, fields::FieldLocation::Cell);

        const std::size_t n_plane = 3 * n_total;
        mgr.add_field<double>("grad_nu", n_plane, fields::FieldLocation::Cell);
        mgr.add_field<double>("mut", n_total, fields::FieldLocation::Cell);
        mgr.add_field<double>("wall_dist", n_total, fields::FieldLocation::Cell);

        // Per-eval scratch
        nu_.assign(n_total, 0.0);
        rho_.assign(n_total, 0.0);
    }

    void append_update_slots(std::vector<double*>& u,
                             std::vector<double*>& prev,
                             std::vector<double*>& stage,
                             std::vector<double*>& res,
                             fields::FieldsManager& mgr,
                             const bool needs_prev_snapshot) {
        u.push_back(mgr.get_required_field_ptr<double>("rho_nu"));
        stage.push_back(mgr.get_required_field_ptr<double>("stage_rho_nu"));
        res.push_back(mgr.get_required_field_ptr<double>("res_rho_nu"));
        if (needs_prev_snapshot) {
            prev.push_back(mgr.get_required_field_ptr<double>("prev_rho_nu"));
        }
    }

    void register_halo(halo::HaloExchanger& halo, fields::FieldsManager& mgr) {
        rho_nu_  = mgr.get_required_field_ptr<double>("rho_nu");
        res_nu_  = mgr.get_required_field_ptr<double>("res_rho_nu");
        grad_nu_ = mgr.get_required_field_ptr<double>("grad_nu");
        mut_     = mgr.get_required_field_ptr<double>("mut");
        dist_    = mgr.get_required_field_ptr<double>("wall_dist");

        // Join the aggregated (single-message-per-neighbour) phases
        std::array<double*, 1> transported = {rho_nu_};
        halo.register_cell_fields(transported);

        std::array<double*, 1> grad_bases = {grad_nu_};
        halo.register_grad_limiters(grad_bases, n_total_, {});
    }

    /** @brief Freestream reference state (from [initial]; call before initialize). */
    void set_freestream_state(const double rho, const double p) noexcept {
        rho_inf_ = rho;
        p_inf_ = p;
    }

    /**
     * @brief Builds per-zone BC treatment and computes the wall distance.
     */
    template <eos::EquationOfState EOS>
    void initialize(const mesh::MeshPart& mp,
                    const BoundaryConfig& bcfg,
                    const EOS& eos,
                    const gradient::VertexAdjacency& adj,
                    halo::HaloExchanger& halo,
                    const MPI_Comm comm) {
        n_own_   = static_cast<std::size_t>(mp.n_own);
        n_cells_ = static_cast<std::size_t>(mp.n_cells);
        n_inner_ = static_cast<std::size_t>(mp.n_inner_faces);
        n_faces_ = static_cast<std::size_t>(mp.n_faces);

        // 1. Mean-flow BC type -> nu_tilde zone treatment (module-owned)
        face_kind_.assign(n_faces_ - n_inner_, NuZoneKind::Symmetry);
        {
            std::vector<NuZoneKind> patch_kind(mp.patches.size(), NuZoneKind::Symmetry);
            for (const auto& desc : bcfg.patches) {
                const std::size_t p = static_cast<std::size_t>(desc.patch_id);
                if (p < patch_kind.size()) {
                    patch_kind[p] = zone_kind(desc.type);
                }
            }
            for (std::size_t f = n_inner_; f < n_faces_; ++f) {
                const std::size_t p = static_cast<std::size_t>(mp.face_patch[f]);
                face_kind_[f - n_inner_] = (p < patch_kind.size())
                                         ? patch_kind[p]
                                         : NuZoneKind::Symmetry;
            }

            // 2. Wall distance from no-slip walls only
            std::vector<bool> wall_patch(mp.patches.size(), false);
            for (std::size_t p = 0; p < wall_patch.size(); ++p) {
                wall_patch[p] = patch_kind[p] == NuZoneKind::Wall;
            }
            solve_wall_distance(mp, adj, halo, comm, wall_patch,
                                max_distance_sweeps, distance_tolerance, dist_);
        }

        // 3. Freestream working values
        const double T_inf = eos.temperature_rhop(rho_inf_, p_inf_);
        nu_inf_ = nu_inf_ratio * physics::ViscousFlow::viscosity(T_inf) / rho_inf_;
        rho_nu_inf_ = rho_inf_ * nu_inf_;
    }

    /** @brief Fills all transported buffers with the freestream value. */
    void init_state(fields::FieldsManager& mgr) const {
        std::fill(rho_nu_, rho_nu_ + n_total_, rho_nu_inf_);
        double* stage = mgr.get_required_field_ptr<double>("stage_rho_nu");
        std::fill(stage, stage + n_total_, rho_nu_inf_);
        if (double* prev = mgr.get_field_ptr<double>("prev_rho_nu"); prev) {
            std::fill(prev, prev + n_total_, rho_nu_inf_);
        }
    }

    // --- Per residual-evaluation hooks (ordered by the solver pipeline) ------

    /** @brief nu_tilde gradient (own LSQ plane) + mirrored ghost planes.
     *         Requires pre_sweep (fills the nu_tilde scratch). */
    void compute_gradients(const gradient::GradientMethod& gm,
                           const mesh::MeshPart& mp) const {
        gm.compute_scalar(nu_.data(), grad_nu_, n_total_);

        // BC ghost planes mirror the interior (sign flip at no-slip walls)
        for (std::size_t f = n_inner_; f < n_faces_; ++f) {
            const std::size_t i = f - n_inner_;
            const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);
            const std::size_t cg = n_cells_ + i;
            const double s = (face_kind_[i] == NuZoneKind::Wall) ? -1.0 : 1.0;

            for (std::size_t p = 0; p < 3; ++p) {
                grad_nu_[p * n_total_ + cg] = s * grad_nu_[p * n_total_ + c0];
            }
        }
    }

    /** @brief Ghost values of rho*nu_tilde on boundary faces. */
    template <eos::EquationOfState EOS>
    void apply_bcs(const EOS& eos, const mesh::MeshPart& mp) const {
        for (std::size_t f = n_inner_; f < n_faces_; ++f) {
            const std::size_t i = f - n_inner_;
            const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);
            const std::size_t cg = n_cells_ + i;

            switch (face_kind_[i]) {
                case NuZoneKind::Wall:
                    rho_nu_[cg] = -rho_nu_[c0];
                    break;
                case NuZoneKind::Inflow: {
                    // Specific nu_tilde is fixed; ghost primitives are complete
                    const double rho_g = eos.density_Tp(q_.tmp[cg], q_.prs[cg]);
                    rho_nu_[cg] = rho_g * nu_inf_;
                    break;
                }
                case NuZoneKind::Outflow:
                case NuZoneKind::Symmetry:
                    rho_nu_[cg] = rho_nu_[c0];
                    break;
            }
        }
    }

    /**
     * @brief Per-eval scratch: nu_tilde and density on owned + MPI ghost cells,
     *        eddy viscosity for the viscous sweep (BC ghosts: mut = 0).
     */
    template <eos::EquationOfState EOS>
    void pre_sweep(const EOS& eos, const mesh::MeshPart& mp) {
        (void)mp;
        for (std::size_t c = 0; c < n_cells_; ++c) {
            const double rho = eos.density_Tp(q_tmp(c), q_prs(c));
            const double nu = rho_nu_[c] / rho;

            rho_[c] = rho;
            nu_[c] = nu;

            const double nu_lam = physics::ViscousFlow::viscosity(q_tmp(c)) / rho;
            const double chi = nu / nu_lam;
            const double chi3 = chi * chi * chi;
            const double fv1 = chi3 / (chi3 + kCv1_3);
            mut_[c] = rho * nu * fv1;
        }

        // BC ghost slots carry no turbulence; exact zeros keep the viscous
        // face averages correct at walls (nu_t wall = 0).
        std::fill(mut_ + n_cells_, mut_ + n_total_, 0.0);
    }

    /**
     * @brief Convection (upwind on the mean-flow mass flux) + diffusion.
     *
     * @param mdot Face mass flux (signed, area-scaled) stored by the NS sweep.
     * @param geom Mean-flow viscous geometry (face displacement vectors).
     */
    template <eos::EquationOfState EOS, typename MeanGeometry>
    void face_sweep(const EOS& eos,
                    const mesh::MeshPart& mp,
                    double* CFD_RESTRICT lam,
                    const double* CFD_RESTRICT mdot,
                    const MeanGeometry& geom) {
        std::fill(res_nu_, res_nu_ + n_total_, 0.0);

        const LocalIndex* CFD_RESTRICT owner = mp.face_owner.data();
        const LocalIndex* CFD_RESTRICT neigh = mp.face_neigh.data();
        const double* CFD_RESTRICT nx = mp.face_normal_x.data();
        const double* CFD_RESTRICT ny = mp.face_normal_y.data();
        const double* CFD_RESTRICT nz = mp.face_normal_z.data();
        const double* CFD_RESTRICT area = mp.face_area.data();

        const double inv_sigma = 1.0 / kSigma;

        // 1. Interior faces
        for (std::size_t f = 0; f < n_inner_; ++f) {
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t c1 = static_cast<std::size_t>(neigh[f]);

            // Upwind convection on the shared mass flux
            const double m = mdot[f];
            const double nu_up = (m >= 0.0) ? nu_[c0] : nu_[c1];
            const double f_conv = m * nu_up;

            // Diffusion: skew-corrected face gradient, face-averaged coefficients
            double gn[3];
            face_gradient(f, c0, nu_[c0], c1, nu_[c1], geom, gn);

            const double Tf = 0.5 * (q_tmp(c0) + q_tmp(c1));
            const double mu_f = physics::ViscousFlow::viscosity(Tf);
            const double rho_f = 0.5 * (rho_[c0] + rho_[c1]);
            const double nu_f = 0.5 * (nu_[c0] + nu_[c1]);
            const double coef = inv_sigma * (mu_f + rho_f * nu_f);

            const double f_diff = coef * (gn[0] * nx[f] + gn[1] * ny[f] + gn[2] * nz[f]) * area[f];

            const double flux = f_conv + f_diff;
            res_nu_[c0] += flux;
            res_nu_[c1] -= flux;

            // Spectral radius: |u_n| A + diffusion estimate [m^3 / s]
            const double visc_term = 2.0 * inv_sigma * (mu_f + rho_f * nu_f)
                                   / rho_f * area[f] * geom.inv_d[f];
            const double w = std::abs(m) / rho_f + visc_term;
            lam[c0] += w;
            lam[c1] += w;
        }

        // 2. Boundary faces: owner + BC ghost
        for (std::size_t f = n_inner_; f < n_faces_; ++f) {
            const std::size_t i = f - n_inner_;
            const std::size_t c0 = static_cast<std::size_t>(owner[f]);
            const std::size_t cg = n_cells_ + i;

            const double m = mdot[f];
            const bool inflow = m < 0.0;
            const double nu_up = inflow ? nu_inf_ : nu_[c0];
            const double f_conv = m * nu_up;

            // Ghost state: specific nu_tilde from the BC ghost slot
            const double rho_g = eos.density_Tp(q_tmp(cg), q_prs(cg));
            const double nu_g = inflow ? nu_inf_ : rho_nu_[cg] / rho_g;

            double gn[3];
            face_gradient(f, c0, nu_[c0], cg, nu_g, geom, gn);

            const double Tf = 0.5 * (q_tmp(c0) + q_tmp(cg));
            const double mu_f = physics::ViscousFlow::viscosity(Tf);
            const double rho_f = 0.5 * (rho_[c0] + rho_g);
            const double coef = inv_sigma * (mu_f + rho_f * 0.5 * (nu_[c0] + nu_g));

            const double f_diff = coef * (gn[0] * nx[f] + gn[1] * ny[f] + gn[2] * nz[f]) * area[f];

            res_nu_[c0] += f_conv + f_diff;
        }
    }

    /**
     * @brief Production / destruction / cb2 source terms on owned cells.
     *
     * @param mean_grad Mean-flow primitive gradients (velocity -> strain rate).
     */
    template <eos::EquationOfState EOS>
    void cell_sources(const EOS& eos,
                      const mesh::MeshPart& mp,
                      const fields::ConstPrimitiveGradView mean_grad) {
        (void)eos;
        const double inv_k2 = 1.0 / (kKappa * kKappa);
        const double cb2_over_sigma = kCb2 / kSigma;

        for (std::size_t c = 0; c < n_own_; ++c) {
            const double rho = rho_[c];
            const double nu = nu_[c];
            const double d = std::max(dist_[c], kMinWallDist);
            const double inv_d2 = 1.0 / (d * d);

            // Strain-rate magnitude S = sqrt(2 S_ij S_ij)
            const double ux = mean_grad.dvx_dx(c), uy = mean_grad.dvx_dy(c), uz = mean_grad.dvx_dz(c);
            const double vx = mean_grad.dvy_dx(c), vy = mean_grad.dvy_dy(c), vz = mean_grad.dvy_dz(c);
            const double wx = mean_grad.dvz_dx(c), wy = mean_grad.dvz_dy(c), wz = mean_grad.dvz_dz(c);

            const double s_xy = uy + vx;
            const double s_xz = uz + wx;
            const double s_yz = vz + wy;
            const double s2 = 2.0 * (ux * ux + vy * vy + wz * wz)
                            + 0.5 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz);
            const double S = std::sqrt(std::max(s2, 0.0));

            const double nu_lam = physics::ViscousFlow::viscosity(q_tmp(c)) / rho;
            const double chi = nu / nu_lam;
            const double chi3 = chi * chi * chi;
            const double fv1 = chi3 / (chi3 + kCv1_3);
            const double fv2 = 1.0 - chi / (1.0 + chi * fv1);

            const double sbar = std::max(S + nu * fv2 * inv_k2 * inv_d2, kSbarFloor);

            double r = nu / (sbar * kKappa * kKappa * d * d);
            r = std::min(std::max(r, 0.0), 10.0);

            const double r2 = r * r;
            const double r6 = r2 * r2 * r2;
            const double g = r + kCw2 * (r6 - r);
            const double g2 = std::max(g * g, kSbarFloor);
            const double g6 = g2 * g2 * g2;
            const double cw3_6 = kCw3 * kCw3 * kCw3 * kCw3 * kCw3 * kCw3;
            // (1 + cw3^6)^(1/6): sqrt of the cube root
            const double fw = g * std::sqrt(std::cbrt((1.0 + cw3_6) / g6));

            const double P = kCb1 * sbar * rho * nu;
            const double D = kCw1 * fw * rho * nu * nu * inv_d2;

            const double gx = grad_nu_[c];
            const double gy = grad_nu_[n_total_ + c];
            const double gz = grad_nu_[2 * n_total_ + c];
            const double cb2_term = cb2_over_sigma * rho * (gx * gx + gy * gy + gz * gz);

            // Residual = outflux - sources: the update adds (P - D + cb2) * dt
            res_nu_[c] += mp.cell_volume[c] * (D - P + cb2_term);
        }
    }

    /** @brief Positivity clamp on the just-updated state slots (per stage). */
    void post_stage(const std::span<double* const> state) const {
        double* CFD_RESTRICT var = state[0];
        for (std::size_t c = 0; c < n_own_; ++c) {
            if (var[c] < 0.0) {
                var[c] = 0.0;
            }
        }
    }

    /** @brief Appends module output fields (arrays live on [0, n_own)). */
    void append_output(std::vector<io::vtk::SolutionField>& out) const {
        out.push_back({"nu_tilde", nu_.data()});
        out.push_back({"mu_t", mut_});
    }

    /** @brief Eddy-viscosity data for the mean-flow viscous sweep. */
    [[nodiscard]] const double* mut_data() const noexcept { return mut_; }

    [[nodiscard]] static NuZoneKind zone_kind(const bc::BCType t) noexcept {
        switch (t) {
            case bc::BCType::NoSlipWall:
            case bc::BCType::NoSlipWallHeatFlux:
                return NuZoneKind::Wall;
            case bc::BCType::SupersonicInlet:
            case bc::BCType::SubsonicInlet:
            case bc::BCType::Farfield:
                return NuZoneKind::Inflow;
            case bc::BCType::SupersonicOutlet:
            case bc::BCType::SubsonicOutlet:
                return NuZoneKind::Outflow;
            case bc::BCType::SlipWall:
            case bc::BCType::Symmetry:
            default:
                return NuZoneKind::Symmetry;
        }
    }

    /**
     * @brief Binds the (stable) mean-flow primitive view for kernel access.
     *        Must be called once after the mean views exist.
     */
    void bind_primitives(fields::ConstPrimitiveView q) noexcept {
        q_ = q;
    }

private:
    [[nodiscard]] double q_tmp(const std::size_t c) const noexcept { return q_.tmp[c]; }
    [[nodiscard]] double q_prs(const std::size_t c) const noexcept { return q_.prs[c]; }

    /** @brief Averaged + skew-corrected nu_tilde face gradient. */
    template <typename MeanGeometry>
    void face_gradient(const std::size_t f,
                       const std::size_t cl,
                       const double nu_l,
                       const std::size_t cr,
                       const double nu_r,
                       const MeanGeometry& geom,
                       double gn[3]) const noexcept {
        const std::size_t s = n_total_;
        const double dx = geom.dx[f];
        const double dy = geom.dy[f];
        const double dz = geom.dz[f];

        gn[0] = 0.5 * (grad_nu_[cl] + grad_nu_[cr]);
        gn[1] = 0.5 * (grad_nu_[s + cl] + grad_nu_[s + cr]);
        gn[2] = 0.5 * (grad_nu_[2 * s + cl] + grad_nu_[2 * s + cr]);

        const double jump = nu_r - nu_l;
        const double gd = gn[0] * dx + gn[1] * dy + gn[2] * dz;
        const double inv_d = geom.inv_d[f];
        const double corr = (jump - gd) * inv_d * inv_d;
        gn[0] += corr * dx;
        gn[1] += corr * dy;
        gn[2] += corr * dz;
    }

    static constexpr double kSbarFloor = 1.0e-10;
    static constexpr double kMinWallDist = 1.0e-10;

    static_assert(physics::PhysicsGeneral<SpalartAllmaras>,
                  "SpalartAllmaras must satisfy physics::PhysicsGeneral");

    // Sizes
    std::size_t n_own_ = 0;
    std::size_t n_cells_ = 0;
    std::size_t n_inner_ = 0;
    std::size_t n_faces_ = 0;
    std::size_t n_total_ = 0;

    // Field pointers (bound in register_halo)
    double* rho_nu_ = nullptr;
    double* res_nu_ = nullptr;
    double* grad_nu_ = nullptr;
    double* mut_ = nullptr;
    double* dist_ = nullptr;

    // Per-eval scratch
    std::vector<double> nu_;   ///< nu_tilde [0, n_cells)
    std::vector<double> rho_;  ///< density   [0, n_cells)
    fields::ConstPrimitiveView q_{}; ///< stable mean primitives (bound once)

    // BC data
    std::vector<NuZoneKind> face_kind_; ///< [n_bfaces]

    // Freestream reference
    double rho_inf_ = constants::kIsaDensity;
    double p_inf_ = constants::kIsaPressure;
    double nu_inf_ = 0.0;
    double rho_nu_inf_ = 0.0;
};

} // namespace cfd::solver::turb

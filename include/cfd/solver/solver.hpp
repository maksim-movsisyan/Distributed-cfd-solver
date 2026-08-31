// Composition root of one solver run.
// EOS, FluxPolicy, ReconPolicy and TimePolicy are compile-time policy types
// (fully inlined); all memory allocation is zero-overhead SoA via FieldsManager
// and FieldsView. State updates run through generic per-variable update blocks
// (fields::block_*), so the time integrator is agnostic to the equation system
// size — physics-module variables will append to the same blocks.
//
// The Solver doubles as the residual OPERATOR handed to TimePolicy::advance
// (public interface: evaluate_residual / compute_dt / *_slots / alpha /
// n_owned / ping_pong).
#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"
#include "cfd/solver/bc/bc_manager.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/solver/fields/fields_manager.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/gradient/lsq_gradient.hpp"
#include "cfd/solver/halo.hpp"
#include "cfd/solver/physics/physics_concepts.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"
#include "cfd/solver/residual_kernel.hpp"
#include "cfd/solver/time/time_policy.hpp"

namespace cfd::solver {

template <eos::EquationOfState EOS, typename FluxPolicy, recon::ReconstructionPolicy ReconPolicy,
          physics::PhysicsGeneral PhysPolicy, template <typename> class TimePolicyT>
class Solver {
public:
    // Concrete time policy bound to this solver instantiation. The injected
    // class name completes TimePolicyT<Solver> without circular instantiation.
    using TimePolicy = TimePolicyT<Solver>;

    // Cell gradients are needed by MUSCL reconstruction OR by the physics
    // (viscous fluxes) — future physics modules compose through this flag too.
    static constexpr bool kNeedsGradients = ReconPolicy::kNeedsGradients
                                         || PhysPolicy::kNeedsGradients;
    static constexpr bool kHasModules = PhysPolicy::kNumExtraVars > 0;

    Solver(const SolverConfig& cfg,
           const BoundaryConfig& bcfg,
           const EOS eos,
           const PhysPolicy phys,
           const mesh::MeshPart& mp,
           const MPI_Comm comm)
        : mp_(mp),
          cfg_(cfg),
          eos_(eos),
          phys_(phys),
          halo_(mp, comm),
          kernel_(mp, eos, phys.base),
          comm_(comm) {

        // 1. Initialize boundary conditions
        bcs_.initialize(bcfg, mp, eos_);

        // 2. Allocate SoA storage in FieldsManager
        const std::size_t n_inner  = static_cast<std::size_t>(mp_.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mp_.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mp_.n_cells);
        const std::size_t n_own    = static_cast<std::size_t>(mp_.n_own);
        const std::size_t n_bfaces = n_faces - n_inner;
        const std::size_t n_total  = n_cells + n_bfaces;

        allocate_fields(n_total);

        // 2b. Physics modules: fields + update slots (appended after the
        //     mean-flow variables, so time integrators advance them jointly)
        if constexpr (kHasModules) {
            phys_.register_fields(mgr_, n_total, TimePolicy::kNeedsPrevSnapshot);
        }

        // 3. Build fast non-owning Views and the update-block slot registry
        bind_views();

        if constexpr (kHasModules) {
            phys_.append_update_slots(u_slots_, prev_slots_, stage_slots_, res_slots_,
                                      mgr_, TimePolicy::kNeedsPrevSnapshot);
        }

        // 4. Register the stable SoA views with the aggregated halo engine
        //    (single message per neighbour per phase; modules join the same
        //    messages)
        register_halo_payloads();
        if constexpr (kHasModules) {
            phys_.register_halo(halo_, mgr_);
            phys_.bind_primitives(q_view_.as_const());
        }

        // 5. Allocate step metrics (+ module face mass flux storage)
        lam_.resize(n_cells, 0.0);
        dt_.resize(n_own, 0.0);
        alpha_.resize(n_own, 0.0);
        if constexpr (PhysPolicy::kNeedsFaceMdot) {
            mdot_.assign(n_faces, 0.0);
        }

        // 6. Gradient method + module initialization (BCs, wall distance) +
        //    reconstruction geometry (mesh-fixed)
        if constexpr (kNeedsGradients) {
            adjacency_ = gradient::build_vertex_adjacency(mp_);
            grad_method_ = std::make_unique<gradient::LsqGradient>(mp_, adjacency_);
        }
        if constexpr (kHasModules) {
            phys_.set_freestream_state(cfg_.init_rho, cfg_.init_p);
            phys_.template initialize<EOS>(mp_, bcfg, eos_, adjacency_, halo_, comm_);
        }
        geom_ = ReconPolicy::build_geometry(mp_, cfg_.limiter_venkat_k);
    }

    int run() {
        static_assert(time::TimeIntegrationPolicy<TimePolicy>,
                      "time policy must satisfy the time::TimeIntegrationPolicy concept");

        init_fields();
        const double wall0 = MPI_Wtime();

        std::array<double, PhysPolicy::kNumVars> norm0{};
        bool have_norm0 = false;
        long long last_iter = 0;

        mpi::log_info("solver: physics=%s flux=%s recon=%s limiter=%s scheme=%s cfl=%.3f max_iter=%lld",
                      PhysPolicy::full_name().c_str(), FluxPolicy::name(), ReconPolicy::name(),
                      ReconPolicy::limiter_name(),
                      TimePolicy::name(),
                      cfg_.cfl, static_cast<long long>(cfg_.max_iterations));

        for (long long iter = 1; iter <= cfg_.max_iterations; ++iter) {
            last_iter = iter;

            // --- One full time step (all update blocks) ---
            time_.advance(*this);

            // --- Diagnostics & convergence ---
            // The global L2 reduction runs only every residual_interval
            // iterations (plus the first and the last): per-iteration
            // collectives would cap strong scaling at high rank counts.
            const bool diagnose = iter == 1
                               || iter % cfg_.residual_interval == 0
                               || iter == cfg_.max_iterations;

            bool converged = false;
            double rel = 1.0;
            if (diagnose) {
                std::array<double, PhysPolicy::kNumVars> l2{};
                residual_norms(l2);

                if (!std::isfinite(l2[0])) {
                    mpi::log_warn_rank("solver: non-finite residual detected, dumping state");
                    write_fields("blowup");
                    return 1;
                }

                if (!have_norm0) {
                    norm0 = l2;
                    have_norm0 = true;
                } else {
                    rel = relative_residual(l2, norm0);
                }

                log_progress(iter, MPI_Wtime() - wall0, l2, rel);
                if (g_verbose >= 1) {
                    log_boundary_integrals();
                }

                converged = rel <= cfg_.residual_tolerance;
            }

            if (cfg_.field_interval > 0 && iter % cfg_.field_interval == 0) {
                write_fields(make_stem("iter", iter));
            }

            if (converged) {
                mpi::log_info("solver: converged at iteration %lld (rel=%.3e)", iter, rel);
                break;
            }
        }

        write_fields(make_stem("final", last_iter));
        mpi::log_info("solver: done in %lld iterations, wall time %.3f s",
                      last_iter, MPI_Wtime() - wall0);
        return 0;
    }

    // --- Residual Operator interface (consumed by TimePolicy) ----------------

    /**
     * @brief Evaluates the full residual pipeline R(state) for the given
     *        update-block state slots (u or a stage buffer).
     */
    void evaluate_residual(const std::span<double* const> state) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mp_.n_own);

        // 1. Owned conservative states -> primitives
        const double* CFD_RESTRICT s0 = state[0];
        const double* CFD_RESTRICT s1 = state[1];
        const double* CFD_RESTRICT s2 = state[2];
        const double* CFD_RESTRICT s3 = state[3];
        const double* CFD_RESTRICT s4 = state[4];
        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[PhysPolicy::kNumVars] = {s0[c], s1[c], s2[c], s3[c], s4[c]};
            eos::conserved_to_primitives_pT(eos_, U_c,
                                            q_view_.prs[c], q_view_.vx[c],
                                            q_view_.vy[c], q_view_.vz[c],
                                            q_view_.tmp[c]);
        }

        // 2. MPI halo exchange on primitives + module variables (one message)
        halo_.exchange_fields();

        // 3. Boundary condition ghosts on the primitive fields, then module
        //    variables (module inflow ghosts read the mean-flow ghost state)
        bcs_.apply_all(q_view_, mp_);
        if constexpr (kHasModules) {
            phys_.apply_bcs(eos_, mp_);
        }

        // 4. Module per-eval scratch (nu_tilde / density / eddy viscosity)
        //    before gradients: module LSQ consumes the nu_tilde scratch
        if constexpr (PhysPolicy::kHasEddyViscosity) {
            phys_.pre_sweep(eos_, mp_);
        }

        // 5. Gradients and limiters over owned cells + BCs + packed MPI exchange
        if constexpr (kNeedsGradients) {
            grad_method_->compute(q_view_.as_const(), grad_view_);
            bcs_.apply_grad_all(q_view_.as_const(), grad_view_, mp_);
            if constexpr (ReconPolicy::kNeedsGradients) {
                ReconPolicy::compute_limiters(mp_, q_view_.as_const(),
                                              grad_view_.as_const(), adjacency_,
                                              geom_, phi_view_);
            }
            if constexpr (kHasModules) {
                phys_.compute_gradients(*grad_method_, mp_);
            }
            halo_.exchange_grad_limiters();
        }

        // 6. Mean-flow flux sweeps on the reconstructed primitive states
        //    (stores the face mass flux for module convection when requested)
        kernel_.apply(q_view_.as_const(), grad_view_.as_const(), phi_view_.as_const(),
                      geom_, res_view_, lam_.data(), mut_ptr(),
                      PhysPolicy::kNeedsFaceMdot ? mdot_.data() : nullptr);

        // 7. Module convection + diffusion (upwind on the shared mass flux)
        if constexpr (PhysPolicy::kNeedsFaceMdot) {
            phys_.face_sweep(eos_, mp_, lam_.data(), mdot_.data(),
                             kernel_.phys_geometry());
        }

        // 8. Module source terms (production / destruction)
        if constexpr (kHasModules) {
            phys_.cell_sources(eos_, mp_, grad_view_.as_const());
        }
    }

    void compute_dt() noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        const double* CFD_RESTRICT lam = lam_.data();
        const double* CFD_RESTRICT vol = mp_.cell_volume.data();
        double* CFD_RESTRICT dt        = dt_.data();
        double* CFD_RESTRICT alpha     = alpha_.data();

        for (std::size_t c = 0; c < n_own; ++c) {
            const double l = std::max(lam[c], constants::kSpectralRadiusFloor);
            dt[c]    = cfg_.cfl * vol[c] / l;
            alpha[c] = cfg_.cfl / l;
        }
    }

    // --- Update-block slots (mean-flow components today, module vars later) ---

    [[nodiscard]] std::span<double* const> u_slots() noexcept { return u_slots_; }
    [[nodiscard]] std::span<double* const> stage_slots() noexcept { return stage_slots_; }
    [[nodiscard]] std::span<double* const> prev_slots() noexcept { return prev_slots_; }
    [[nodiscard]] std::span<double* const> res_slots() noexcept { return res_slots_; }

    [[nodiscard]] const double* alpha() const noexcept { return alpha_.data(); }
    [[nodiscard]] std::size_t n_owned() const noexcept {
        return static_cast<std::size_t>(mp_.n_own);
    }

    /** @brief Swaps the primary and stage state slots (ping-pong buffers). */
    void ping_pong() noexcept { std::swap(u_slots_, stage_slots_); }

    /** @brief Module positivity clamps on the buffer a stage just wrote. */
    void post_stage(const std::span<double* const> state) noexcept {
        if constexpr (kHasModules) {
            phys_.post_stage(state);
        }
    }

private:
    // --- Memory Allocation, Views & Slot Binding ------------------------------

    void allocate_fields(const std::size_t n_total) {
        // Primary State U & Q
        mgr_.add_field<double>("rho",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhou", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhov", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhow", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhoE", n_total, fields::FieldLocation::Cell);

        mgr_.add_field<double>("prs", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vx",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vy",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vz",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("tmp", n_total, fields::FieldLocation::Cell);

        // Residuals
        mgr_.add_field<double>("res1", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res2", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res3", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res4", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res5", n_total, fields::FieldLocation::Cell);

        // Gradients and limiters (allocated if reconstruction OR physics need them)
        if constexpr (kNeedsGradients) {
            const std::size_t n_plane = 3 * n_total;
            mgr_.add_field<double>("grad_prs", n_plane, fields::FieldLocation::Cell);
            mgr_.add_field<double>("grad_vx",  n_plane, fields::FieldLocation::Cell);
            mgr_.add_field<double>("grad_vy",  n_plane, fields::FieldLocation::Cell);
            mgr_.add_field<double>("grad_vz",  n_plane, fields::FieldLocation::Cell);
            mgr_.add_field<double>("grad_tmp", n_plane, fields::FieldLocation::Cell);

            mgr_.add_field<double>("phi_prs", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phi_vx",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phi_vy",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phi_vz",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phi_tmp", n_total, fields::FieldLocation::Cell);
        }

        // RK scratch: stage buffer always, u^n snapshot only for multistage schemes
        if constexpr (TimePolicy::kNeedsPrevSnapshot) {
            mgr_.add_field<double>("prev_rho",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhou", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhov", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhow", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhoE", n_total, fields::FieldLocation::Cell);
        }

        mgr_.add_field<double>("stage_rho",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhou", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhov", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhow", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhoE", n_total, fields::FieldLocation::Cell);
    }

    void bind_views() {
        q_view_ = {
            mgr_.get_required_field_ptr<double>("prs"),
            mgr_.get_required_field_ptr<double>("vx"),
            mgr_.get_required_field_ptr<double>("vy"),
            mgr_.get_required_field_ptr<double>("vz"),
            mgr_.get_required_field_ptr<double>("tmp")
        };

        res_view_ = {
            mgr_.get_required_field_ptr<double>("res1"),
            mgr_.get_required_field_ptr<double>("res2"),
            mgr_.get_required_field_ptr<double>("res3"),
            mgr_.get_required_field_ptr<double>("res4"),
            mgr_.get_required_field_ptr<double>("res5")
        };

        // Update-block slots: one entry per solvable variable per role. The
        // u/stage vectors are swapped by ping_pong(); u_view() always reflects
        // the current primary state.
        u_slots_ = {
            mgr_.get_required_field_ptr<double>("rho"),
            mgr_.get_required_field_ptr<double>("rhou"),
            mgr_.get_required_field_ptr<double>("rhov"),
            mgr_.get_required_field_ptr<double>("rhow"),
            mgr_.get_required_field_ptr<double>("rhoE")
        };

        stage_slots_ = {
            mgr_.get_required_field_ptr<double>("stage_rho"),
            mgr_.get_required_field_ptr<double>("stage_rhou"),
            mgr_.get_required_field_ptr<double>("stage_rhov"),
            mgr_.get_required_field_ptr<double>("stage_rhow"),
            mgr_.get_required_field_ptr<double>("stage_rhoE")
        };

        if constexpr (TimePolicy::kNeedsPrevSnapshot) {
            prev_slots_ = {
                mgr_.get_required_field_ptr<double>("prev_rho"),
                mgr_.get_required_field_ptr<double>("prev_rhou"),
                mgr_.get_required_field_ptr<double>("prev_rhov"),
                mgr_.get_required_field_ptr<double>("prev_rhow"),
                mgr_.get_required_field_ptr<double>("prev_rhoE")
            };
        }

        res_slots_ = {
            mgr_.get_required_field_ptr<double>("res1"),
            mgr_.get_required_field_ptr<double>("res2"),
            mgr_.get_required_field_ptr<double>("res3"),
            mgr_.get_required_field_ptr<double>("res4"),
            mgr_.get_required_field_ptr<double>("res5")
        };

        if constexpr (kNeedsGradients) {
            const std::size_t n_total = mgr_.get_field_size("rho");
            grad_view_ = {
                n_total,
                mgr_.get_required_field_ptr<double>("grad_prs"),
                mgr_.get_required_field_ptr<double>("grad_vx"),
                mgr_.get_required_field_ptr<double>("grad_vy"),
                mgr_.get_required_field_ptr<double>("grad_vz"),
                mgr_.get_required_field_ptr<double>("grad_tmp")
            };
            phi_view_ = {
                mgr_.get_required_field_ptr<double>("phi_prs"),
                mgr_.get_required_field_ptr<double>("phi_vx"),
                mgr_.get_required_field_ptr<double>("phi_vy"),
                mgr_.get_required_field_ptr<double>("phi_vz"),
                mgr_.get_required_field_ptr<double>("phi_tmp")
            };
        }
    }

    /** @brief Conservative view of the current primary state (5 mean-flow vars). */
    [[nodiscard]] fields::ConservativeView<double> u_view() const noexcept {
        return {u_slots_[0], u_slots_[1], u_slots_[2], u_slots_[3], u_slots_[4]};
    }

    void register_halo_payloads() {
        // Mean-flow primitives join the aggregated fields phase
        std::array<double*, 5> q_fields = {
            q_view_.prs, q_view_.vx, q_view_.vy, q_view_.vz, q_view_.tmp
        };
        halo_.register_cell_fields(q_fields);

        // Mean-flow gradients + limiters join the aggregated gradient phase
        if constexpr (kNeedsGradients) {
            std::array<double*, 5> grad_bases = {
                grad_view_.prs_grad, grad_view_.vx_grad, grad_view_.vy_grad,
                grad_view_.vz_grad, grad_view_.tmp_grad
            };
            std::array<double*, 5> lims = {
                phi_view_.prs, phi_view_.vx, phi_view_.vy, phi_view_.vz, phi_view_.tmp
            };
            halo_.register_grad_limiters(grad_bases, grad_view_.stride, lims);
        }
    }

    void init_fields() {
        double U[PhysPolicy::kNumVars];
        eos::primitives_rhop_to_conserved(eos_,
                                          cfg_.init_rho,
                                          cfg_.init_velocity[0],
                                          cfg_.init_velocity[1],
                                          cfg_.init_velocity[2],
                                          cfg_.init_p,
                                          U);

        auto u = u_view();
        const std::size_t n_total = mgr_.get_field_size("rho");
        for (std::size_t c = 0; c < n_total; ++c) {
            u.rho[c]  = U[0];
            u.rhou[c] = U[1];
            u.rhov[c] = U[2];
            u.rhow[c] = U[3];
            u.rhoE[c] = U[4];
        }

        if constexpr (kHasModules) {
            phys_.init_state(mgr_);
        }
    }

    // --- Diagnostics & VTU Output -------------------------------------------

    void residual_norms(std::array<double, PhysPolicy::kNumVars>& l2) const noexcept {
        std::array<double, PhysPolicy::kNumVars> local{};
        const std::size_t n_own = static_cast<std::size_t>(mp_.n_own);

        for (std::size_t c = 0; c < n_own; ++c) {
            local[0] += res_view_.res1[c] * res_view_.res1[c];
            local[1] += res_view_.res2[c] * res_view_.res2[c];
            local[2] += res_view_.res3[c] * res_view_.res3[c];
            local[3] += res_view_.res4[c] * res_view_.res4[c];
            local[4] += res_view_.res5[c] * res_view_.res5[c];
        }

        MPI_Allreduce(local.data(), l2.data(), PhysPolicy::kNumVars, MPI_DOUBLE, MPI_SUM, comm_);
        const double scale = 1.0 / static_cast<double>(std::max<GlobalIndex>(mp_.n_cells_g, 1));
        for (auto& val : l2) {
            val = std::sqrt(val * scale);
        }
    }

    static double relative_residual(const std::array<double, PhysPolicy::kNumVars>& l2,
                                     const std::array<double, PhysPolicy::kNumVars>& norm0) noexcept {
        double rel = 0.0;
        for (std::size_t v = 0; v < l2.size(); ++v) {
            rel = std::max(rel, l2[v] / std::max(norm0[v], constants::kResidualNormFloor));
        }
        return rel;
    }

    void log_progress(const long long iter, const double wall,
                      const std::array<double, PhysPolicy::kNumVars>& l2,
                      const double rel) const {
        const double mom = std::sqrt(l2[1] * l2[1] + l2[2] * l2[2] + l2[3] * l2[3]);
        double dt_min = 0.0;
        double dt_max = 0.0;
        if (!dt_.empty()) {
            dt_min = mpi::d_min(*std::min_element(dt_.begin(), dt_.end()));
            dt_max = mpi::d_max(*std::max_element(dt_.begin(), dt_.end()));
        }
        mpi::log_info("iter %lld/%lld  wall %7.2fs  L2[mass]=%.4e  L2[mom]=%.4e  L2[energy]=%.4e  rel=%.3e  dt=[%.3e,%.3e]",
                      iter, static_cast<long long>(cfg_.max_iterations),
                      wall, l2[0], mom, l2[4], rel, dt_min, dt_max);
    }

    void refresh_primitives_for_audit() {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        auto u = u_view();

        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[PhysPolicy::kNumVars] = {
                u.rho[c], u.rhou[c], u.rhov[c], u.rhow[c], u.rhoE[c]
            };
            eos::conserved_to_primitives_pT(eos_, U_c,
                                            q_view_.prs[c], q_view_.vx[c],
                                            q_view_.vy[c], q_view_.vz[c],
                                            q_view_.tmp[c]);
        }
        halo_.exchange_fields();
        bcs_.apply_all(q_view_, mp_);
        if constexpr (kHasModules) {
            phys_.apply_bcs(eos_, mp_);
        }

        if constexpr (kNeedsGradients) {
            grad_method_->compute(q_view_.as_const(), grad_view_);
            bcs_.apply_grad_all(q_view_.as_const(), grad_view_, mp_);
            if constexpr (ReconPolicy::kNeedsGradients) {
                ReconPolicy::compute_limiters(mp_, q_view_.as_const(),
                                              grad_view_.as_const(), adjacency_,
                                              geom_, phi_view_);
            }
            if constexpr (kHasModules) {
                phys_.compute_gradients(*grad_method_, mp_);
            }
        }
    }

    void log_boundary_integrals() {
        std::vector<double> mass;
        std::vector<double> energy;

        refresh_primitives_for_audit();
        kernel_.boundary_integrals(q_view_.as_const(), grad_view_.as_const(),
                                   phi_view_.as_const(), geom_, mass, energy, mut_ptr());

        const auto n = static_cast<int>(mass.size());
        std::vector<double> gmass(static_cast<std::size_t>(n));
        std::vector<double> genergy(static_cast<std::size_t>(n));

        MPI_Allreduce(mass.data(), gmass.data(), n, MPI_DOUBLE, MPI_SUM, comm_);
        MPI_Allreduce(energy.data(), genergy.data(), n, MPI_DOUBLE, MPI_SUM, comm_);

        for (std::size_t p = 0; p < gmass.size(); ++p) {
            mpi::log_stat("  patch %lld '%s': mass flux %+.6e kg/s, energy flux %+.6e W",
                          static_cast<long long>(p),
                          mp_.patches[p].name.c_str(),
                          gmass[p], genergy[p]);
        }
    }

    static std::string make_stem(const char* tag, const long long iter) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%09lld", tag, iter);
        return std::string(buf);
    }

    void write_fields(const std::string& stem) const {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        std::vector<double> rho(n_own), vx(n_own), vy(n_own), vz(n_own), pr(n_own), mach(n_own);

        const auto u = u_view();
        for (std::size_t c = 0; c < n_own; ++c) {
            const double U[PhysPolicy::kNumVars] = {
                u.rho[c], u.rhou[c], u.rhov[c], u.rhow[c], u.rhoE[c]
            };
            const double r = U[0];
            const double p = eos::pressure(eos_, U);
            const double a = eos_.sound_speed_rhop(r, p);

            rho[c]  = r;
            vx[c]   = U[1] / r;
            vy[c]   = U[2] / r;
            vz[c]   = U[3] / r;
            pr[c]   = p;
            mach[c] = std::sqrt(U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) / (r * a);
        }

        const io::vtk::SolutionField mean_fields[] = {
            {"rho",      rho.data()},
            {"u",        vx.data()},
            {"v",        vy.data()},
            {"w",        vz.data()},
            {"pressure", pr.data()},
            {"mach",     mach.data()}
        };
        std::vector<io::vtk::SolutionField> fields(mean_fields,
                                                   mean_fields + sizeof(mean_fields) / sizeof(mean_fields[0]));

        // Module outputs are zero-copy views of live module arrays
        if constexpr (kHasModules) {
            phys_.append_output(fields);
        }

        io::vtk::write_solution_vtu(mp_, fields.data(),
                                    static_cast<int>(fields.size()),
                                    cfg_.output_dir, stem, comm_);
    }

    // --- State Members ------------------------------------------------------

    const mesh::MeshPart& mp_;
    SolverConfig cfg_;
    EOS eos_;
    PhysPolicy phys_;
    bc::BoundaryManager<EOS> bcs_;
    halo::HaloExchanger halo_;

    using MeanFlowPhys = typename PhysPolicy::BaseType;
    ResidualKernel<EOS, FluxPolicy, ReconPolicy, MeanFlowPhys> kernel_;
    
    TimePolicy time_{};
    MPI_Comm comm_{MPI_COMM_WORLD};

    gradient::VertexAdjacency adjacency_;
    std::unique_ptr<gradient::GradientMethod> grad_method_;
    typename ReconPolicy::Geometry geom_;

    fields::FieldsManager mgr_;
    fields::PrimitiveView<double> q_view_{};
    fields::ResidualView<double> res_view_{};
    fields::PrimitiveGradView<double> grad_view_{};
    fields::PrimitiveView<double> phi_view_{};

    // Update-block slot registry: [variable][role] pointer table
    std::vector<double*> u_slots_;
    std::vector<double*> prev_slots_;
    std::vector<double*> stage_slots_;
    std::vector<double*> res_slots_;

    std::vector<double> lam_;   ///< Per-cell spectral radius [0, n_cells)
    std::vector<double> dt_;    ///< Local time step [0, n_own)
    std::vector<double> alpha_; ///< dt / Volume [0, n_own)
    std::vector<double> mdot_;  ///< Face mass flux (module convection) [0, n_faces)

    /** @brief Eddy-viscosity data for the viscous sweep (nullptr w/o modules). */
    [[nodiscard]] const double* mut_ptr() const noexcept {
        if constexpr (PhysPolicy::kHasEddyViscosity) {
            return phys_.mut_data();
        }
        return nullptr;
    }
};

int run_solver(const SolverConfig& cfg,
               const BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               MPI_Comm comm);

} // namespace cfd::solver

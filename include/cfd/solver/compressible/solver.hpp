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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"
#include "cfd/fields/halo.hpp"
#include "cfd/fields/fields_manager.hpp"
#include "cfd/numerics/gradient/gradient_manager.hpp"
#include "cfd/numerics/reconstruction/concepts.hpp"
#include "cfd/bc/config.hpp"

#include "cfd/solver/compressible/bc_manager.hpp"
#include "cfd/solver/compressible/config.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"
#include "cfd/solver/compressible/eos/state_conversions.hpp"
#include "cfd/solver/compressible/physics/physics_concepts.hpp"
#include "cfd/solver/compressible/residual_kernel.hpp"
#include "cfd/solver/compressible/jacobian_kernel.hpp"
#include "cfd/solver/compressible/time/concepts.hpp"
#include "cfd/solver/time_mode.hpp"

namespace cfd::solver::compressible {

template <eos::EquationOfStatePolicy EOS, typename FluxPolicy, typename ReconPolicy,
          physics::PhysicsGeneral PhysPolicy, template <typename> class TimePolicyT, typename TimeMode = SteadyMode>
          requires numerics::recon::ReconstructionPolicy<ReconPolicy, PhysPolicy::kNumVars>
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

    static constexpr mesh::AuxGeomType kAuxGeometry = ReconPolicy::kAuxGeometry|
                                                      PhysPolicy::kAuxGeometry|
                                                      TimePolicy::kAuxGeometry; 
    static constexpr mesh::AuxConnType kAuxConnectivity = ReconPolicy::kAuxConnectivity|
                                                          PhysPolicy::kAuxConnectivity|
                                                          TimePolicy::kAuxConnectivity; 

    static constexpr bool kNeedsMatrix = TimePolicy::kNeedsMatrix;

    Solver(const SolverConfig& cfg,
           const bc::BoundaryConfig& bcfg,
           const EOS& eos,
           const PhysPolicy& phys,
           const mesh::MeshPart& mesh,
           const MPI_Comm comm)
        : mesh_(mesh),
          aux_geom_(),
          aux_conn_(),
          cfg_(cfg),
          eos_(eos),
          phys_(phys),
          residual_kernel_(mesh, aux_conn_, aux_geom_, eos, phys),
          jacobian_kernel_(mesh, aux_conn_, aux_geom_, eos, phys),
          bcs_(),
          halo_(mesh, comm),
          comm_(comm) {

        // 1. Initialize boundary condition patches
        bcs_.initialize(bcfg, mesh, eos_);

        // 2. Allocate SoA storage in FieldsManager
        const std::size_t n_inner  = static_cast<std::size_t>(mesh.n_inner_faces);
        const std::size_t n_faces  = static_cast<std::size_t>(mesh.n_faces);
        const std::size_t n_cells  = static_cast<std::size_t>(mesh.n_cells);
        const std::size_t n_own    = static_cast<std::size_t>(mesh.n_own);
        const std::size_t n_bfaces = n_faces - n_inner;
        const std::size_t n_total  = n_cells + n_bfaces;

        allocate_fields(n_total);

        // 2b. Physics modules variable registration
        if constexpr (kHasModules) {
            phys_.register_fields(mgr_, n_total);
        }

        // 3. Bind SoA pointer slots
        bind_slots();

        if constexpr (kHasModules) {
            phys_.append_update_slots(u_slots_, stage_slots_, res_slots_, mgr_);
        }

        // 4. Register stable SoA buffers with MPI halo exchanger
        register_halo_payloads();
        if constexpr (kHasModules) {
            phys_.register_halo(halo_, mgr_);
            phys_.bind_primitives(q_const_slots_);
        }

        // 5. Allocate step metrics (+ module face mass flux storage)
        lam_.resize(n_cells, 0.0);
        dt_.resize(n_own, 0.0);
        alpha_.resize(n_own, 0.0);
        if constexpr (PhysPolicy::kNeedsFaceMdot) {
            mdot_.assign(n_faces, 0.0);
        }

        // 6. Build dual connectivity and auxiliary geometry
        aux_conn_.add_connectivity(mesh, kAuxConnectivity);
        aux_geom_.add_geometry(mesh, kAuxGeometry);

        // 7. Gradient manager and physics initialization
        if constexpr (kNeedsGradients) {
            grad_mgr_.create_gradient(cfg.gradient); 
            grad_mgr_.setup_gradient(mesh, aux_conn_);
        }
        if constexpr (kHasModules) {
            phys_.set_freestream_state(cfg_.init_rho, cfg_.init_p);
            phys_.template initialize<EOS>(mesh, aux_conn_, aux_geom_, bcfg, eos_, halo_, comm_);
        }

        // 8. Implicit solver linear system assembly setup
        if constexpr (kNeedsMatrix) {
            time_.system_setup(mesh, aux_conn_, cfg.linear_solver_params, comm_);
        }
    }

    int run() {
        static_assert(time::TimeIntegrationPolicy<TimePolicy>,
                      "Time policy must satisfy the time::TimeIntegrationPolicy concept");

        init_fields();
        if constexpr (TimeMode::kIsUnsteady) {
            return run_dual_time();
        } else {
            return run_steady();
        }
    }

    // --- Residual Operator interface (consumed by TimePolicy) ----------------

    /**
     * @brief Evaluates spatial residual R(state) for the active update-block slots.
     */
    void evaluate_residual(const std::span<double* const> state) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);

        // 1. Convert owned cell conservative state to primitives
        const double* CFD_RESTRICT s0 = state[0];
        const double* CFD_RESTRICT s1 = state[1];
        const double* CFD_RESTRICT s2 = state[2];
        const double* CFD_RESTRICT s3 = state[3];
        const double* CFD_RESTRICT s4 = state[4];

        double* CFD_RESTRICT q_p = q_slots_[0];
        double* CFD_RESTRICT q_u = q_slots_[1];
        double* CFD_RESTRICT q_v = q_slots_[2];
        double* CFD_RESTRICT q_w = q_slots_[3];
        double* CFD_RESTRICT q_T = q_slots_[4];
        
        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[PhysPolicy::kNumVars] = {s0[c], s1[c], s2[c], s3[c], s4[c]};
            eos::conserved_to_primitives_pT(eos_, U_c, q_p[c], q_u[c], q_v[c], q_w[c], q_T[c]);
        }

        // 2. MPI halo exchange on primitives + module variables (one message)
        halo_.exchange_fields();

        // 3. Boundary condition ghosts on the primitive fields, then module
        //    variables (module inflow ghosts read the mean-flow ghost state)
        bcs_.update_ghost_cells(q_slots_, mesh_);
        if constexpr (kHasModules) {
            phys_.apply_bcs(eos_, mesh_);
        }

        // 4. Module per-eval scratch (nu_tilde / density / eddy viscosity)
        //    before gradients: module LSQ consumes the nu_tilde scratch
        if constexpr (PhysPolicy::kHasEddyViscosity) {
            phys_.pre_sweep(eos_, mesh_);
        }

        // 5. Gradients and limiters over owned cells + BCs + packed MPI exchange
        if constexpr (kNeedsGradients) {
            grad_mgr_.apply_gradient_set(q_const_slots_, gx_slots_, gy_slots_, gz_slots_, mesh_, aux_conn_);
            bcs_.update_ghost_cells_grad(q_const_slots_, gx_slots_, gy_slots_, gz_slots_, mesh_);

            if constexpr (ReconPolicy::kNeedsGradients) {
                ReconPolicy::template compute_limiters<PhysPolicy::kNumVars>(
                    mesh_, aux_conn_, 
                    q_const_slots_.data(),
                    gx_const_slots_.data(), 
                    gy_const_slots_.data(), 
                    gz_const_slots_.data(),
                    phi_slots_.data(), 
                    cfg_.limiter_venkat_k);
            }
            if constexpr (kHasModules) {
                phys_.compute_gradients(grad_mgr_, mesh_);
            }
            halo_.exchange_grad_limiters();
        }

        // 6. Mean-flow flux sweeps on the reconstructed primitive states
        //    (stores the face mass flux for module convection when requested)
        residual_kernel_.apply(q_const_slots_, gx_const_slots_, gy_const_slots_, gz_const_slots_,
                               phi_const_slots_, res_slots_, lam_.data(), mut_ptr(),
                               PhysPolicy::kNeedsFaceMdot ? mdot_.data() : nullptr);

        // 7. Auxiliary module face sweeps (upwinded on convective mass flux)
        if constexpr (PhysPolicy::kNeedsFaceMdot) {
            phys_.face_sweep(eos_, mesh_, aux_conn_, aux_geom_, lam_.data(), mdot_.data());
        }

        // 8. Auxiliary module volumetric source terms
        if constexpr (kHasModules) {
            phys_.cell_sources(eos_, mesh_, gx_const_slots_, gy_const_slots_, gz_const_slots_);
        }

        // 9. Unsteady BDF pseudo-source contribution
        if constexpr (TimeMode::kIsUnsteady) {
            add_unsteady_source_term(state);
        }
    }

    void assemble_jacobian(const LocalIndex* CFD_RESTRICT row_ptr,
                           const LocalIndex* CFD_RESTRICT cols,
                           const LocalIndex* CFD_RESTRICT diag_idx,
                           double* CFD_RESTRICT values) {
        jacobian_kernel_.apply(q_const_slots_, row_ptr, cols, diag_idx, values, alpha_.data(), mut_ptr());
    }

    void compute_dt() noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);
        const double* CFD_RESTRICT lam = lam_.data();
        const double* CFD_RESTRICT vol = mesh_.cell_volume.data();
        double* CFD_RESTRICT dt        = dt_.data();
        double* CFD_RESTRICT alpha     = alpha_.data();

        if constexpr (kNeedsMatrix) {
            //Implicit: alpha = Volume / dt = lambda / CFL
            const double inv_cfl = 1.0 / cfg_.cfl;
            
            double unsteady_diag_coeff = 0.0;
            if constexpr (TimeMode::kIsUnsteady) {
                const double c0 = (bdf_order_cur_ == 2) ? 1.5 : 1.0;
                unsteady_diag_coeff = c0 / cfg_.dt;
            }
            
            for (std::size_t c = 0; c < n_own; ++c) {
                const double l = std::max(lam[c], constants::kSpectralRadiusFloor);
                dt[c]    = cfg_.cfl * vol[c] / l;
                alpha[c] = (l * inv_cfl) + (vol[c] * unsteady_diag_coeff);                  // take into account unstedy terms in jacobian
            }
        } else {
            // Explicit: alpha = dt / Volume = CFL / lambda  
            for (std::size_t c = 0; c < n_own; ++c) {
                const double l = std::max(lam[c], constants::kSpectralRadiusFloor);
                const double cfl_over_l = cfg_.cfl / l;
                dt[c]    = cfl_over_l * vol[c];
                alpha[c] = cfl_over_l;
            }
        }
    }

    // --- Update-block slots (mean-flow components today, module vars later) ---

    [[nodiscard]] std::span<double* const> u_slots() noexcept { return u_slots_; }
    [[nodiscard]] std::span<double* const> stage_slots() noexcept { return stage_slots_; }
    [[nodiscard]] std::span<double* const> res_slots() noexcept { return res_slots_; }

    [[nodiscard]] const double* alpha() const noexcept { return alpha_.data(); }
    [[nodiscard]] std::size_t n_owned() const noexcept { return static_cast<std::size_t>(mesh_.n_own); }

    /** @brief Swaps the primary and stage state slots (ping-pong buffers). */
    void ping_pong() noexcept { std::swap(u_slots_, stage_slots_); }

    /** @brief Module positivity clamps on the buffer a stage just wrote. */
    void post_stage(const std::span<double* const> state) noexcept {
        if constexpr (kHasModules) {
            phys_.post_stage(state);
        }
    }

    // --- Implicit-scheme services (consumed by implicit time policies) --------

    [[nodiscard]] const mesh::MeshPart& mesh() const noexcept { return mesh_; }
    [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return comm_; }
    [[nodiscard]] const SolverConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] const double* local_dt() const noexcept { return dt_.data(); }

private:
    // --- Memory Allocation, Views & Slot Binding ------------------------------
    void allocate_fields(const std::size_t n_total) {
        // Conservative state variables
        mgr_.add_field<double>("rho",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhou", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhov", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhow", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("rhoE", n_total, fields::FieldLocation::Cell);

        // Primitive variables
        mgr_.add_field<double>("prs", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vx",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vy",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("vz",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("tmp", n_total, fields::FieldLocation::Cell);

        // Residual accumulators
        mgr_.add_field<double>("res1", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res2", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res3", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res4", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("res5", n_total, fields::FieldLocation::Cell);

        // Gradient components and limiters
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

        mgr_.add_field<double>("stage_rho",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhou", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhov", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhow", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhoE", n_total, fields::FieldLocation::Cell);

        // Physical time level snapshots for BDF time integration
        if constexpr (TimeMode::kIsUnsteady) {
            mgr_.add_field<double>("phys_un_rho",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phys_un_rhou", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phys_un_rhov", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phys_un_rhow", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("phys_un_rhoE", n_total, fields::FieldLocation::Cell);

            if constexpr (TimeMode::kBdfOrder == 2) {
                mgr_.add_field<double>("phys_unm1_rho",  n_total, fields::FieldLocation::Cell);
                mgr_.add_field<double>("phys_unm1_rhou", n_total, fields::FieldLocation::Cell);
                mgr_.add_field<double>("phys_unm1_rhov", n_total, fields::FieldLocation::Cell);
                mgr_.add_field<double>("phys_unm1_rhow", n_total, fields::FieldLocation::Cell);
                mgr_.add_field<double>("phys_unm1_rhoE", n_total, fields::FieldLocation::Cell);
            }
        }
    }

    void bind_slots() {
        // Primitives SoA
        q_slots_ = {
            mgr_.get_required_field_ptr<double>("prs"),
            mgr_.get_required_field_ptr<double>("vx"),
            mgr_.get_required_field_ptr<double>("vy"),
            mgr_.get_required_field_ptr<double>("vz"),
            mgr_.get_required_field_ptr<double>("tmp")
        };
        q_const_slots_.assign(q_slots_.begin(), q_slots_.end());

        // Residuals SoA
        res_slots_ = {
            mgr_.get_required_field_ptr<double>("res1"),
            mgr_.get_required_field_ptr<double>("res2"),
            mgr_.get_required_field_ptr<double>("res3"),
            mgr_.get_required_field_ptr<double>("res4"),
            mgr_.get_required_field_ptr<double>("res5")
        };

        // Primary state SoA
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

        if constexpr (kNeedsGradients) {
            const std::size_t n_total = mgr_.get_field_size("rho");
            
            std::array<double*, 5> grad_slots_ = {
                mgr_.get_required_field_ptr<double>("grad_prs"),
                mgr_.get_required_field_ptr<double>("grad_vx"),
                mgr_.get_required_field_ptr<double>("grad_vy"),
                mgr_.get_required_field_ptr<double>("grad_vz"),
                mgr_.get_required_field_ptr<double>("grad_tmp")
            };

            gx_slots_ = {
                grad_slots_[0],
                grad_slots_[1],
                grad_slots_[2],
                grad_slots_[3],
                grad_slots_[4]
            };
            gy_slots_ = {
                grad_slots_[0] + n_total,
                grad_slots_[1] + n_total,
                grad_slots_[2] + n_total,
                grad_slots_[3] + n_total,
                grad_slots_[4] + n_total
            };
            gz_slots_ = {
                grad_slots_[0] + 2 * n_total,
                grad_slots_[1] + 2 * n_total,
                grad_slots_[2] + 2 * n_total,
                grad_slots_[3] + 2 * n_total,
                grad_slots_[4] + 2 * n_total
            };

            gx_const_slots_.assign(gx_slots_.begin(), gx_slots_.end());
            gy_const_slots_.assign(gy_slots_.begin(), gy_slots_.end());
            gz_const_slots_.assign(gz_slots_.begin(), gz_slots_.end());

            phi_slots_ = {
                mgr_.get_required_field_ptr<double>("phi_prs"),
                mgr_.get_required_field_ptr<double>("phi_vx"),
                mgr_.get_required_field_ptr<double>("phi_vy"),
                mgr_.get_required_field_ptr<double>("phi_vz"),
                mgr_.get_required_field_ptr<double>("phi_tmp")
            };
            phi_const_slots_.assign(phi_slots_.begin(), phi_slots_.end());
        }

        if constexpr (TimeMode::kIsUnsteady) {
            phys_un_slots_ = {
                mgr_.get_required_field_ptr<double>("phys_un_rho"),
                mgr_.get_required_field_ptr<double>("phys_un_rhou"),
                mgr_.get_required_field_ptr<double>("phys_un_rhov"),
                mgr_.get_required_field_ptr<double>("phys_un_rhow"),
                mgr_.get_required_field_ptr<double>("phys_un_rhoE")
            };

            if constexpr (TimeMode::kBdfOrder == 2) {
                phys_unm1_slots_ = {
                    mgr_.get_required_field_ptr<double>("phys_unm1_rho"),
                    mgr_.get_required_field_ptr<double>("phys_unm1_rhou"),
                    mgr_.get_required_field_ptr<double>("phys_unm1_rhov"),
                    mgr_.get_required_field_ptr<double>("phys_unm1_rhow"),
                    mgr_.get_required_field_ptr<double>("phys_unm1_rhoE")
                };
            }
        }
    }

    void register_halo_payloads() {
        std::array<double*, 5> q_fields = {
            q_slots_[0], q_slots_[1], q_slots_[2], q_slots_[3], q_slots_[4]
        };
        halo_.register_cell_fields(q_fields);

        if constexpr (kNeedsGradients) {
            std::array<double*, 5> grad_bases_x = {
                gx_slots_[0], gx_slots_[1], gx_slots_[2], gx_slots_[3], gx_slots_[4]
            };
            std::array<double*, 5> grad_bases_y = {
                gy_slots_[0], gy_slots_[1], gy_slots_[2], gy_slots_[3], gy_slots_[4]
            };
            std::array<double*, 5> grad_bases_z = {
                gz_slots_[0], gz_slots_[1], gz_slots_[2], gz_slots_[3], gz_slots_[4]
            };
            std::array<double*, 5> lims = {
                phi_slots_[0], phi_slots_[1], phi_slots_[2], phi_slots_[3], phi_slots_[4]
            };
            halo_.register_grad_limiters(grad_bases_x, grad_bases_y, grad_bases_z, lims);
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

        const std::size_t n_total = mgr_.get_field_size("rho");
        for (std::size_t c = 0; c < n_total; ++c) {
            u_slots_[0][c] = U[0];
            u_slots_[1][c] = U[1];
            u_slots_[2][c] = U[2];
            u_slots_[3][c] = U[3];
            u_slots_[4][c] = U[4];

            if constexpr (TimeMode::kIsUnsteady) {
                phys_un_slots_[0][c] = U[0];
                phys_un_slots_[1][c] = U[1];
                phys_un_slots_[2][c] = U[2];
                phys_un_slots_[3][c] = U[3];
                phys_un_slots_[4][c] = U[4];
                if constexpr (TimeMode::kBdfOrder == 2) {
                    phys_unm1_slots_[0][c] = U[0];
                    phys_unm1_slots_[1][c] = U[1];
                    phys_unm1_slots_[2][c] = U[2];
                    phys_unm1_slots_[3][c] = U[3];
                    phys_unm1_slots_[4][c] = U[4];
                }
            }
        }

        if constexpr (kHasModules) {
            phys_.init_state(mgr_);
        }
    }

    void add_unsteady_source_term(const std::span<double* const> state) noexcept {
        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);
        const double* CFD_RESTRICT vol = mesh_.cell_volume.data();
        const double inv_dt = 1.0 / cfg_.dt;

        if (bdf_order_cur_ == 2) {
            const double c0 = 1.5 * inv_dt;
            const double c1 = -2.0 * inv_dt;
            const double c2 = 0.5 * inv_dt;

            for (std::size_t v = 0; v < PhysPolicy::kNumVars; ++v) {
                const double* CFD_RESTRICT u_cur = state[v];
                const double* CFD_RESTRICT u_n   = phys_un_slots_[v];
                const double* CFD_RESTRICT u_nm1 = phys_unm1_slots_[v];
                double* CFD_RESTRICT res         = res_slots_[v];

                for (std::size_t c = 0; c < n_own; ++c) {
                    res[c] += vol[c] * (c0 * u_cur[c] + c1 * u_n[c] + c2 * u_nm1[c]);
                }
            }
        } else {
            const double c0 = 1.0 * inv_dt;
            const double c1 = -1.0 * inv_dt;

            for (std::size_t v = 0; v < PhysPolicy::kNumVars; ++v) {
                const double* CFD_RESTRICT u_cur = state[v];
                const double* CFD_RESTRICT u_n   = phys_un_slots_[v];
                double* CFD_RESTRICT res         = res_slots_[v];

                for (std::size_t c = 0; c < n_own; ++c) {
                    res[c] += vol[c] * (c0 * u_cur[c] + c1 * u_n[c]);
                }
            }
        }
    }

    void shift_physical_time_levels() noexcept {
        if constexpr (TimeMode::kIsUnsteady) {
            const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own); 
            const std::size_t n_vars = PhysPolicy::kNumVars;

            if constexpr (TimeMode::kBdfOrder == 2) {
                std::swap(phys_unm1_slots_, phys_un_slots_);
            }

            for (std::size_t v = 0; v < n_vars; ++v) {
                std::copy_n(u_slots_[v], n_own, phys_un_slots_[v]); 
            }
        }
    }

    int run_steady() {
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

            time_.advance(*this);

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

    int run_dual_time() {
        const double wall0 = MPI_Wtime();
        mpi::log_info("solver (unsteady): physics=%s flux=%s recon=%s limiter=%s scheme=%s dt=%.3e total_steps=%lld max_subiter=%lld",
                    PhysPolicy::full_name().c_str(), FluxPolicy::name(), ReconPolicy::name(),
                    ReconPolicy::limiter_name(),
                    TimePolicy::name(),
                    cfg_.dt, static_cast<long long>(cfg_.max_time_steps),
                    static_cast<long long>(cfg_.max_iterations));
        
        double t_phys = 0.0;

        for (std::int64_t phys_step = 1; phys_step <= cfg_.max_time_steps; ++phys_step) {
            t_phys += cfg_.dt;
            bdf_order_cur_ = (phys_step == 1 || TimeMode::kBdfOrder == 1) ? 1 : 2;

            std::array<double, PhysPolicy::kNumVars> norm0{};
            bool sub_converged = false;
            bool have_norm0 = false;
            std::int64_t last_subiter = 0;
            
            for (std::int64_t subiter = 1; subiter <= cfg_.max_iterations; ++subiter) {
                last_subiter = subiter;
                time_.advance(*this);
                
                const bool diagnose = subiter == 1
                                   || subiter % 5 == 0
                                   || subiter == cfg_.max_iterations;

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

                    converged = rel <= cfg_.residual_tolerance;
                }

                if (converged) {
                    sub_converged = true;
                    break;
                }
            }

            // U^{n-1} = U^n, U^n = U^{n+1}
            shift_physical_time_levels();

            if (cfg_.field_interval > 0 && phys_step % cfg_.field_interval == 0) {
                write_fields(make_stem("unsteady", phys_step));
            }

            mpi::log_info("Time step %lld/%lld: t = %.5e s | subiters: %lld/%lld (%s)", 
                          static_cast<long long>(phys_step), static_cast<long long>(cfg_.max_time_steps),
                          t_phys, 
                          static_cast<long long>(last_subiter), static_cast<long long>(cfg_.max_iterations),
                          sub_converged ? "converged" : "limit reached");
        }

        write_fields(make_stem("final", cfg_.max_time_steps));
        mpi::log_info("solver: unsteady calculation finished in %lld steps, wall time %.3f s",
                      static_cast<long long>(cfg_.max_time_steps), MPI_Wtime() - wall0);
        return 0;
    }

    // --- Diagnostics & VTU Output -------------------------------------------

    void residual_norms(std::array<double, PhysPolicy::kNumVars>& l2) const noexcept {
        std::array<double, PhysPolicy::kNumVars> local{};
        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);

        for (std::size_t v = 0; v < PhysPolicy::kNumVars; ++v) {
            const double* CFD_RESTRICT res_v = res_slots_[v];
            for (std::size_t c = 0; c < n_own; ++c) {
                local[v] += res_v[c] * res_v[c];
            }
        }

        MPI_Allreduce(local.data(), l2.data(), PhysPolicy::kNumVars, MPI_DOUBLE, MPI_SUM, comm_);
        const double scale = 1.0 / static_cast<double>(std::max<GlobalIndex>(mesh_.n_cells_g, 1));
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
        const std::size_t n_own = static_cast<std::size_t>(mesh_.n_own);
        const double* CFD_RESTRICT u0 = u_slots_[0];
        const double* CFD_RESTRICT u1 = u_slots_[1];
        const double* CFD_RESTRICT u2 = u_slots_[2];
        const double* CFD_RESTRICT u3 = u_slots_[3];
        const double* CFD_RESTRICT u4 = u_slots_[4];

        double* CFD_RESTRICT q_p = q_slots_[0];
        double* CFD_RESTRICT q_u = q_slots_[1];
        double* CFD_RESTRICT q_v = q_slots_[2];
        double* CFD_RESTRICT q_w = q_slots_[3];
        double* CFD_RESTRICT q_T = q_slots_[4];

        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[PhysPolicy::kNumVars] = {u0[c], u1[c], u2[c], u3[c], u4[c]};
            eos::conserved_to_primitives_pT(eos_, U_c, q_p[c], q_u[c], q_v[c], q_w[c], q_T[c]);
        }
        
        halo_.exchange_fields();
        bcs_.update_ghost_cells(q_slots_, mesh_);
        if constexpr (kHasModules) {
            phys_.apply_bcs(eos_, mesh_);
        }

        if constexpr (kNeedsGradients) {
            grad_mgr_.apply_gradient_set(q_const_slots_, gx_slots_, gy_slots_, gz_slots_, mesh_, aux_conn_);
            bcs_.update_ghost_cells_grad(q_const_slots_, gx_slots_, gy_slots_, gz_slots_, mesh_);

            if constexpr (ReconPolicy::kNeedsGradients) {
                ReconPolicy::template compute_limiters<PhysPolicy::kNumVars>(
                    mesh_, aux_conn_, 
                    q_const_slots_.data(),
                    gx_const_slots_.data(), 
                    gy_const_slots_.data(), 
                    gz_const_slots_.data(),
                    phi_slots_.data(), 
                    cfg_.limiter_venkat_k);
            }
            if constexpr (kHasModules) {
                phys_.compute_gradients(grad_mgr_, mesh_);
            }
        }
    }

    void log_boundary_integrals() {
        std::vector<double> mass;
        std::vector<double> energy;

        refresh_primitives_for_audit();
        residual_kernel_.boundary_integrals(q_const_slots_, gx_const_slots_, gy_const_slots_, gz_const_slots_,
                                            phi_const_slots_, mass, energy, mut_ptr());

        const auto n = static_cast<int>(mass.size());
        std::vector<double> gmass(static_cast<std::size_t>(n));
        std::vector<double> genergy(static_cast<std::size_t>(n));

        MPI_Allreduce(mass.data(), gmass.data(), n, MPI_DOUBLE, MPI_SUM, comm_);
        MPI_Allreduce(energy.data(), genergy.data(), n, MPI_DOUBLE, MPI_SUM, comm_);

        for (std::size_t p = 0; p < gmass.size(); ++p) {
            mpi::log_stat("  patch %lld '%s': mass flux %+.6e kg/s, energy flux %+.6e W",
                          static_cast<long long>(p),
                          mesh_.patches[p].name.c_str(),
                          gmass[p], genergy[p]);
        }
    }

    static std::string make_stem(const char* tag, const long long iter) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%09lld", tag, iter);
        return std::string(buf);
    }

    void write_fields(const std::string& stem) const {
        const auto n_own = static_cast<std::size_t>(mesh_.n_own);
        std::vector<double> rho(n_own), vx(n_own), vy(n_own), vz(n_own), pr(n_own), mach(n_own);

        const double* CFD_RESTRICT u0 = u_slots_[0];
        const double* CFD_RESTRICT u1 = u_slots_[1];
        const double* CFD_RESTRICT u2 = u_slots_[2];
        const double* CFD_RESTRICT u3 = u_slots_[3];
        const double* CFD_RESTRICT u4 = u_slots_[4];

        for (std::size_t c = 0; c < n_own; ++c) {
            const double U[PhysPolicy::kNumVars] = {u0[c], u1[c], u2[c], u3[c], u4[c]};
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

        if constexpr (kHasModules) {
            phys_.append_output(fields);
        }

        io::vtk::write_solution_vtu(mesh_, fields.data(),
                                    static_cast<int>(fields.size()),
                                    cfg_.output_dir, stem, comm_);
    }

    // --- State Members ------------------------------------------------------

    const mesh::MeshPart& mesh_;
    mesh::MeshAuxGeometry aux_geom_;
    mesh::MeshAuxConnectivity aux_conn_;
    SolverConfig cfg_;
    EOS eos_;
    PhysPolicy phys_;
    ResidualKernel<EOS, FluxPolicy, ReconPolicy, PhysPolicy> residual_kernel_;
    JacobianKernel<EOS, FluxPolicy, PhysPolicy> jacobian_kernel_;
    BoundaryManager<EOS> bcs_;
    fields::halo::HaloExchanger halo_;

    TimePolicy time_{};
    MPI_Comm comm_{MPI_COMM_WORLD};

    numerics::gradient::GradientManager grad_mgr_;

    fields::FieldsManager mgr_;

    // Update-block slot registries: [variable][role]
    std::vector<double*> q_slots_;
    std::vector<const double*> q_const_slots_;

    std::vector<double*> gx_slots_, gy_slots_, gz_slots_;
    std::vector<const double*> gx_const_slots_, gy_const_slots_, gz_const_slots_;

    std::vector<double*> phi_slots_;
    std::vector<const double*> phi_const_slots_;

    std::vector<double*> u_slots_;
    std::vector<double*> phys_un_slots_, phys_unm1_slots_;
    std::size_t bdf_order_cur_ = 1;
    std::vector<double*> stage_slots_;
    std::vector<double*> res_slots_;

    std::vector<double> lam_;   ///< Per-cell spectral radius [0, n_cells)
    std::vector<double> dt_;    ///< Local time step [0, n_own)
    std::vector<double> alpha_; ///< CFL diagonal scale [0, n_own)
    std::vector<double> mdot_;  ///< Face convective mass flux [0, n_faces)

    /** @brief Eddy-viscosity data for the viscous sweep (nullptr w/o modules). */
    [[nodiscard]] const double* mut_ptr() const noexcept {
        if constexpr (PhysPolicy::kHasEddyViscosity) {
            return phys_.mut_data();
        }
        return nullptr;
    }
};

int run_solver(const SolverConfig& cfg,
               const bc::BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               MPI_Comm comm);

} // namespace cfd::solver

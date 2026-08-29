// Composition root of one solver run.
// EOS, FluxPolicy, and ReconPolicy are compile-time policy types (fully inlined);
// all memory allocation is zero-overhead SoA via FieldsManager and FieldsView.
#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
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
#include "cfd/solver/eos/ideal_gas.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/solver/fields/fields_manager.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/gradient/gradient.hpp"
#include "cfd/solver/gradient/lsq_gradient.hpp"
#include "cfd/solver/halo.hpp"
#include "cfd/solver/reconstruction/reconstruction.hpp"
#include "cfd/solver/residual_kernel.hpp"

namespace cfd::solver {

template <eos::EquationOfState EOS, typename FluxPolicy, recon::ReconstructionPolicy ReconPolicy>
class Solver {
public:
    Solver(const SolverConfig& cfg,
           const BoundaryConfig& bcfg,
           const EOS eos,
           const mesh::MeshPart& mp,
           const MPI_Comm comm)
        : mp_(mp),
          cfg_(cfg),
          eos_(eos),
          halo_(mp, comm),
          kernel_(mp, eos),
          comm_(comm) {

        // 1. Initialize boundary conditions
        bcs_.initialize(bcfg, mp, eos_.gamma(), eos_.gas_constant());

        // 2. Allocate SoA storage in FieldsManager
        const auto n_inner  = static_cast<std::size_t>(mp_.n_inner_faces);
        const auto n_faces  = static_cast<std::size_t>(mp_.n_faces);
        const auto n_cells  = static_cast<std::size_t>(mp_.n_cells);
        const auto n_own    = static_cast<std::size_t>(mp_.n_own);
        const auto n_bfaces = n_faces - n_inner;
        const auto n_total  = n_cells + n_bfaces;

        allocate_fields(n_total);

        // 3. Build fast non-owning Views
        bind_views();

        // 4. Allocate step metrics
        lam_.resize(n_cells, 0.0);
        inv_vol_.resize(n_own, 0.0);
        dt_.resize(n_own, 0.0);
        alpha_.resize(n_own, 0.0);

        for (std::size_t c = 0; c < n_own; ++c) {
            inv_vol_[c] = 1.0 / mp_.cell_volume[c];
        }

        // 5. Gradient method + reconstruction geometry (mesh-fixed)
        if constexpr (ReconPolicy::kNeedsGradients) {
            adjacency_ = gradient::build_vertex_adjacency(mp_);
            grad_method_ = std::make_unique<gradient::LsqGradient>(mp_, adjacency_);
        }
        geom_ = ReconPolicy::build_geometry(mp_, cfg_.limiter_venkat_k);
    }

    int run() {
        init_fields();
        const double wall0 = MPI_Wtime();

        std::array<double, eos::kNumVars> norm0{};
        bool have_norm0 = false;
        long long last_iter = 0;

        mpi::log_info("solver: flux=%s recon=%s limiter=%s scheme=%s cfl=%.3f max_iter=%lld",
                      FluxPolicy::name(), ReconPolicy::name(),
                      ReconPolicy::limiter_name(),
                      cfg_.scheme == TimeScheme::SspRk3 ? "SSP_RK3" : "FORWARD_EULER",
                      cfg_.cfl, static_cast<long long>(cfg_.max_iterations));

        for (long long iter = 1; iter <= cfg_.max_iterations; ++iter) {
            last_iter = iter;

            if (cfg_.scheme == TimeScheme::ForwardEuler) {
                evaluate_residual(u_view_);
                compute_dt();
                sub_axpy_owned(stage_view_, u_view_, alpha_.data(), res_view_);
                std::swap(u_view_, stage_view_);
            } else {
                // SSP-RK3 (Shu-Osher 3-stage TVD Runge-Kutta)
                // Stage 1: u(1) = u^n - dt/V * R(u^n)
                evaluate_residual(u_view_);
                compute_dt();
                copy_owned(prev_view_, u_view_);
                sub_axpy_owned(stage_view_, u_view_, alpha_.data(), res_view_);

                // Stage 2: u(2) = 3/4 u^n + 1/4 (u(1) - dt/V * R(u(1)))
                evaluate_residual(stage_view_);
                ssp_combine(u_view_, 0.75, prev_view_, 0.25, stage_view_, alpha_.data(), res_view_);

                // Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u(2) - dt/V * R(u(2)))
                evaluate_residual(u_view_);
                ssp_combine(stage_view_, 1.0 / 3.0, prev_view_, 2.0 / 3.0, u_view_, alpha_.data(), res_view_);
                std::swap(u_view_, stage_view_);
            }

            // --- Diagnostics & Convergence ---
            std::array<double, eos::kNumVars> l2{};
            residual_norms(l2);

            if (!std::isfinite(l2[0])) {
                mpi::log_warn_rank("solver: non-finite residual detected, dumping state");
                write_fields("blowup");
                return 1;
            }

            double rel = 1.0;
            if (!have_norm0) {
                norm0 = l2;
                have_norm0 = true;
            } else {
                rel = relative_residual(l2, norm0);
            }

            if (iter == 1 || iter % cfg_.residual_interval == 0) {
                log_progress(iter, MPI_Wtime() - wall0, l2, rel);
                if (g_verbose >= 1) {
                    log_boundary_integrals();
                }
            }

            if (cfg_.field_interval > 0 && iter % cfg_.field_interval == 0) {
                write_fields(make_stem("iter", iter));
            }

            if (rel <= cfg_.residual_tolerance) {
                mpi::log_info("solver: converged at iteration %lld (rel=%.3e)", iter, rel);
                break;
            }
        }

        write_fields(make_stem("final", last_iter));
        mpi::log_info("solver: done in %lld iterations, wall time %.3f s",
                      last_iter, MPI_Wtime() - wall0);
        return 0;
    }

private:
    // --- Residual Evaluation Pipeline ---------------------------------------

    void evaluate_residual(fields::ConservativeView<double> u) noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);

        // 1. Owned conservative states -> primitives
        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[eos::kNumVars] = {
                u.rho[c], u.rhou[c], u.rhov[c], u.rhow[c], u.rhoE[c]
            };
            eos::conserved_to_primitives_pT(eos_, U_c,
                                            q_view_.prs[c], q_view_.vx[c],
                                            q_view_.vy[c], q_view_.vz[c],
                                            q_view_.tmp[c]);
        }

        // 2. MPI halo exchange on primitives (ghost cells, one hop)
        halo_.exchange(q_view_);

        // 3. Boundary condition ghosts on the primitive fields
        bcs_.apply_all(q_view_, mp_);

        // 4. Gradients and limiters over owned cells + packed exchange
        if constexpr (ReconPolicy::kNeedsGradients) {
            grad_method_->compute(q_view_.as_const(), grad_view_);
            ReconPolicy::compute_limiters(mp_, q_view_.as_const(),
                                          grad_view_.as_const(), adjacency_,
                                          geom_, phi_view_);
            halo_.exchange_grad_limiter(grad_view_, phi_view_);
        }

        // 5. Numerical flux sweeps on the reconstructed primitive states
        kernel_.apply(q_view_.as_const(), grad_view_.as_const(), phi_view_.as_const(),
                      geom_, res_view_, lam_.data());
    }

    void compute_dt() noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        const double* CFD_RESTRICT lam = lam_.data();
        const double* CFD_RESTRICT vol = mp_.cell_volume.data();
        double* CFD_RESTRICT dt        = dt_.data();
        double* CFD_RESTRICT alpha     = alpha_.data();

        for (std::size_t c = 0; c < n_own; ++c) {
            const double l = std::max(lam[c], 1.0e-14);
            dt[c]    = cfg_.cfl * vol[c] / l;
            alpha[c] = cfg_.cfl / l;
        }
    }

    // --- High-Performance Owned Field Vector Algebra -----------------------

    static void copy_owned(fields::ConservativeView<double>& dst,
                           const fields::ConservativeView<const double>& src,
                           const std::size_t n_own) noexcept {
        std::copy(src.rho,  src.rho  + n_own, dst.rho);
        std::copy(src.rhou, src.rhou + n_own, dst.rhou);
        std::copy(src.rhov, src.rhov + n_own, dst.rhov);
        std::copy(src.rhow, src.rhow + n_own, dst.rhow);
        std::copy(src.rhoE, src.rhoE + n_own, dst.rhoE);
    }

    void copy_owned(fields::ConservativeView<double>& dst,
                    const fields::ConservativeView<double>& src) const noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        copy_owned(dst, src.as_const(), n_own);
    }

    // dst = x - alpha * r
    void sub_axpy_owned(fields::ConservativeView<double>& dst,
                        const fields::ConservativeView<double>& x,
                        const double* CFD_RESTRICT alpha,
                        const fields::ResidualView<double>& r) const noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        for (std::size_t c = 0; c < n_own; ++c) {
            const double a = alpha[c];
            dst.rho[c]  = x.rho[c]  - a * r.res1[c];
            dst.rhou[c] = x.rhou[c] - a * r.res2[c];
            dst.rhov[c] = x.rhov[c] - a * r.res3[c];
            dst.rhow[c] = x.rhow[c] - a * r.res4[c];
            dst.rhoE[c] = x.rhoE[c] - a * r.res5[c];
        }
    }

    // dst = c1 * x + c2 * (y - alpha * r)
    void ssp_combine(fields::ConservativeView<double>& dst,
                     const double c1,
                     const fields::ConservativeView<double>& x,
                     const double c2,
                     const fields::ConservativeView<double>& y,
                     const double* CFD_RESTRICT alpha,
                     const fields::ResidualView<double>& r) const noexcept {
        const auto n_own = static_cast<std::size_t>(mp_.n_own);
        for (std::size_t c = 0; c < n_own; ++c) {
            const double a = alpha[c];
            dst.rho[c]  = c1 * x.rho[c]  + c2 * (y.rho[c]  - a * r.res1[c]);
            dst.rhou[c] = c1 * x.rhou[c] + c2 * (y.rhou[c] - a * r.res2[c]);
            dst.rhov[c] = c1 * x.rhov[c] + c2 * (y.rhov[c] - a * r.res3[c]);
            dst.rhow[c] = c1 * x.rhow[c] + c2 * (y.rhow[c] - a * r.res4[c]);
            dst.rhoE[c] = c1 * x.rhoE[c] + c2 * (y.rhoE[c] - a * r.res5[c]);
        }
    }

    // --- Memory Allocation & Binding ----------------------------------------

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

        // Gradients and limiters (allocated only if 2nd-order reconstruction)
        if constexpr (ReconPolicy::kNeedsGradients) {
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

        // RK3 Previous Snapshot
        if (cfg_.scheme == TimeScheme::SspRk3) {
            mgr_.add_field<double>("prev_rho",  n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhou", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhov", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhow", n_total, fields::FieldLocation::Cell);
            mgr_.add_field<double>("prev_rhoE", n_total, fields::FieldLocation::Cell);
        }

        // Scratch / stage buffer
        mgr_.add_field<double>("stage_rho",  n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhou", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhov", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhow", n_total, fields::FieldLocation::Cell);
        mgr_.add_field<double>("stage_rhoE", n_total, fields::FieldLocation::Cell);
    }

    void bind_views() {
        u_view_ = {
            mgr_.get_required_field_ptr<double>("rho"),
            mgr_.get_required_field_ptr<double>("rhou"),
            mgr_.get_required_field_ptr<double>("rhov"),
            mgr_.get_required_field_ptr<double>("rhow"),
            mgr_.get_required_field_ptr<double>("rhoE")
        };

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

        stage_view_ = {
            mgr_.get_required_field_ptr<double>("stage_rho"),
            mgr_.get_required_field_ptr<double>("stage_rhou"),
            mgr_.get_required_field_ptr<double>("stage_rhov"),
            mgr_.get_required_field_ptr<double>("stage_rhow"),
            mgr_.get_required_field_ptr<double>("stage_rhoE")
        };

        if (cfg_.scheme == TimeScheme::SspRk3) {
            prev_view_ = {
                mgr_.get_required_field_ptr<double>("prev_rho"),
                mgr_.get_required_field_ptr<double>("prev_rhou"),
                mgr_.get_required_field_ptr<double>("prev_rhov"),
                mgr_.get_required_field_ptr<double>("prev_rhow"),
                mgr_.get_required_field_ptr<double>("prev_rhoE")
            };
        }

        if constexpr (ReconPolicy::kNeedsGradients) {
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

    void init_fields() {
        double U[eos::kNumVars];
        eos::primitives_rhop_to_conserved(eos_,
                                          cfg_.init_rho,
                                          cfg_.init_velocity[0],
                                          cfg_.init_velocity[1],
                                          cfg_.init_velocity[2],
                                          cfg_.init_p,
                                          U);

        const auto n_total = mgr_.get_field_size("rho");
        for (std::size_t c = 0; c < n_total; ++c) {
            u_view_.rho[c]  = U[0];
            u_view_.rhou[c] = U[1];
            u_view_.rhov[c] = U[2];
            u_view_.rhow[c] = U[3];
            u_view_.rhoE[c] = U[4];
        }
    }

    // --- Diagnostics & VTU Output -------------------------------------------

    void residual_norms(std::array<double, eos::kNumVars>& l2) const noexcept {
        std::array<double, eos::kNumVars> local{};
        const auto n_own = static_cast<std::size_t>(mp_.n_own);

        for (std::size_t c = 0; c < n_own; ++c) {
            local[0] += res_view_.res1[c] * res_view_.res1[c];
            local[1] += res_view_.res2[c] * res_view_.res2[c];
            local[2] += res_view_.res3[c] * res_view_.res3[c];
            local[3] += res_view_.res4[c] * res_view_.res4[c];
            local[4] += res_view_.res5[c] * res_view_.res5[c];
        }

        MPI_Allreduce(local.data(), l2.data(), eos::kNumVars, MPI_DOUBLE, MPI_SUM, comm_);
        const double scale = 1.0 / static_cast<double>(std::max<GlobalIndex>(mp_.n_cells_g, 1));
        for (auto& val : l2) {
            val = std::sqrt(val * scale);
        }
    }

    static double relative_residual(const std::array<double, eos::kNumVars>& l2,
                                     const std::array<double, eos::kNumVars>& norm0) noexcept {
        double rel = 0.0;
        for (std::size_t v = 0; v < l2.size(); ++v) {
            rel = std::max(rel, l2[v] / std::max(norm0[v], 1.0e-300));
        }
        return rel;
    }

    void log_progress(const long long iter, const double wall,
                      const std::array<double, eos::kNumVars>& l2,
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

        for (std::size_t c = 0; c < n_own; ++c) {
            const double U_c[eos::kNumVars] = {
                u_view_.rho[c], u_view_.rhou[c], u_view_.rhov[c], u_view_.rhow[c], u_view_.rhoE[c]
            };
            eos::conserved_to_primitives_pT(eos_, U_c,
                                            q_view_.prs[c], q_view_.vx[c],
                                            q_view_.vy[c], q_view_.vz[c],
                                            q_view_.tmp[c]);
        }
        halo_.exchange(q_view_);
        bcs_.apply_all(q_view_, mp_);

        if constexpr (ReconPolicy::kNeedsGradients) {
            grad_method_->compute(q_view_.as_const(), grad_view_);
            ReconPolicy::compute_limiters(mp_, q_view_.as_const(),
                                          grad_view_.as_const(), adjacency_,
                                          geom_, phi_view_);
        }
    }

    void log_boundary_integrals() {
        std::vector<double> mass;
        std::vector<double> energy;

        refresh_primitives_for_audit();
        kernel_.boundary_integrals(q_view_.as_const(), grad_view_.as_const(),
                                   phi_view_.as_const(), geom_, mass, energy);

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

        for (std::size_t c = 0; c < n_own; ++c) {
            const double U[eos::kNumVars] = {
                u_view_.rho[c], u_view_.rhou[c], u_view_.rhov[c], u_view_.rhow[c], u_view_.rhoE[c]
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

        const io::vtk::SolutionField fields[] = {
            {"rho",      rho.data()},
            {"u",        vx.data()},
            {"v",        vy.data()},
            {"w",        vz.data()},
            {"pressure", pr.data()},
            {"mach",     mach.data()}
        };

        io::vtk::write_solution_vtu(mp_, fields,
                                    static_cast<int>(sizeof(fields) / sizeof(fields[0])),
                                    cfg_.output_dir, stem, comm_);
    }

    // --- State Members ------------------------------------------------------

    const mesh::MeshPart& mp_;
    SolverConfig cfg_;
    EOS eos_;
    bc::BoundaryManager bcs_;
    halo::HaloExchanger halo_;
    ResidualKernel<EOS, FluxPolicy, ReconPolicy> kernel_;
    MPI_Comm comm_{MPI_COMM_WORLD};

    gradient::VertexAdjacency adjacency_;
    std::unique_ptr<gradient::GradientMethod> grad_method_;
    typename ReconPolicy::Geometry geom_;

    fields::FieldsManager mgr_;
    fields::ConservativeView<double> u_view_{};
    fields::PrimitiveView<double> q_view_{};
    fields::ResidualView<double> res_view_{};
    fields::ConservativeView<double> prev_view_{};
    fields::ConservativeView<double> stage_view_{};
    fields::PrimitiveGradView<double> grad_view_{};
    fields::PrimitiveView<double> phi_view_{};

    std::vector<double> lam_;     ///< Per-cell spectral radius [0, n_cells)
    std::vector<double> inv_vol_; ///< 1.0 / cell_volume [0, n_own)
    std::vector<double> dt_;      ///< Local time step [0, n_own)
    std::vector<double> alpha_;   ///< dt / Volume [0, n_own)
};

int run_solver(const SolverConfig& cfg,
               const BoundaryConfig& bcfg,
               const mesh::MeshPart& mp,
               MPI_Comm comm);

} // namespace cfd::solver
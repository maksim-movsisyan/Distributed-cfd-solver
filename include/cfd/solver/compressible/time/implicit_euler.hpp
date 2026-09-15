#pragma once

#include <sys/types.h>
#include <memory>

#include "cfd/core/types.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/solver/bsr_linear_system.hpp"
#include "cfd/mesh/aux_connectivity.hpp"
#include "cfd/mesh/aux_geometry.hpp"
#include "cfd/linalg/config.hpp"

namespace cfd::solver::compressible::time {

/**
 * @class BackwardEuler
 * @brief Implicit Euler integrator for the mean flow over the linalg module.
 * @tparam Op Residual operator (the Solver instantiation).
 */
template <typename Op>
class BackwardEuler {
public:
    using Operator = Op;

    static constexpr bool kNeedsMatrix = true;
    static constexpr mesh::AuxConnType kAuxConnectivity = mesh::AuxConnType::CellCellsByFace;   // dual graph for matrix initialization
    static constexpr mesh::AuxGeomType kAuxGeometry = mesh::AuxGeomType::None;
    static constexpr const char* name() noexcept { return "BACKWARD_EULER"; }

    void system_setup(const mesh::MeshPart& mesh, const mesh::MeshAuxConnectivity& aux_conn, const linalg::SolverParams& solver_params, MPI_Comm comm) {
        system_ = std::make_unique<BSRLinearSystem<5>>(mesh, aux_conn, solver_params, comm);
    }

    void advance(Op& op) noexcept {
        op.evaluate_residual(op.u_slots());
        op.compute_dt();

        op.assemble_jacobian(system_->matrix().rowPtr().data(),
                             system_->matrix().cols().data(),
                             system_->matrix().diagIndex().data(),
                             system_->matrix().valuesData());

        set_rhs_and_zero_du(op.res_slots());

        if (!system_ -> solve()) {
            mpi::log_stat("BACKWARD_EULER: linear solve stopped does not reach target tolerance");
        }

        const double* CFD_RESTRICT du = system_->du_data();
        const auto u = op.u_slots();
        const auto stage = op.stage_slots();

        for (std::size_t v = 0; v < 5; ++v) {
            double* CFD_RESTRICT dst = stage[v];
            const double* CFD_RESTRICT src = u[v];
            const std::size_t n_own = system_->n_own();

            for (std::size_t c = 0; c < n_own; ++c) {
                dst[c] = src[c] + du[c * 5 + v];
            }
        }

        if constexpr (Op::kHasModules) {
            for (std::size_t v = 5; v < u.size(); ++v) {
                double* CFD_RESTRICT dst = stage[v];
                const double* CFD_RESTRICT src = u[v];
                const std::size_t n_own = system_->n_own();
                for (std::size_t c = 0; c < n_own; ++c) {
                    dst[c] = src[c];
                }
            }
        }

        op.post_stage(op.stage_slots());
        op.ping_pong();
    }

private:
    std::unique_ptr<BSRLinearSystem<5>> system_;

    void set_rhs_and_zero_du(std::span<double* const> res_slots) {
        double* CFD_RESTRICT b = system_->rhs_data();
        double* CFD_RESTRICT du = system_->du_data();
        const std::size_t n_own = system_->n_own();

        for (std::size_t c = 0; c < n_own; ++c) {
            for (std::size_t v = 0; v < 5; ++v) {
                b[c * 5 + v] = -res_slots[v][c];
                du[c * 5 + v] = 0.0;
            }
        }
    }


};

}  // namespace cfd::solver::time

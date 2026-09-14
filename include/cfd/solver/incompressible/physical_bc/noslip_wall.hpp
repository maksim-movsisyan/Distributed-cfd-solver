#pragma once

#include <cassert>
#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/bc/physical/physical_bc.hpp"
#include "cfd/bc/physical/noslip_wall.hpp"
#include "cfd/solver/incompressible/physical_bc/incompressible_bc.hpp"

namespace cfd::solver::incompressible::physical_bc {

/**
 * @class NoSlipWallBC
 * @brief Solid No-Slip Wall boundary condition (stationary or moving).
 */
class NoSlipWallBC final : public IncompressibleBC {
public:
    NoSlipWallBC(std::string zone,
                 const LocalIndex fbeg,
                 const LocalIndex fend,
                 const bc::physical::NoSlipWallParams& p)
        : IncompressibleBC(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() == 4 && "NoSlipWallBC requires 4 variables [p, u, v, w]");
        bc::physical::kernels::no_slip_wall_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 4 && gx.size() == 4 && gy.size() == 4 && gz.size() == 4);
        assert(q.size() == gx.size() && gx.size() == gy.size() && gy.size() == gz.size());

        bc::physical::kernels::no_slip_wall_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
    }

    void apply_momentum_bc(std::span<double*> diag_u,
                           std::span<double*> rhs,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs);
        static_cast<void>(diag_u);
    }

    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const override {
        static_cast<void>(mesh);
        static_cast<void>(rhs_p);
        static_cast<void>(diag_p);
    }

    [[nodiscard]] bc::physical::BCType kind() const noexcept override { 
        return bc::physical::BCType::NoSlipWall; 
    }

private:
    bc::physical::NoSlipWallParams m_p;
};

} // namespace cfd::solver::incompressible::physical_bc
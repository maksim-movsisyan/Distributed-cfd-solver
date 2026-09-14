#pragma once

#include <cassert>
#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/bc/physical/physical_bc.hpp"
#include "cfd/bc/physical/subsonic_outlet.hpp"
#include "cfd/solver/incompressible/physical_bc/incompressible_bc.hpp"

namespace cfd::solver::incompressible::physical_bc {

/**
 * @class PressureOutletBC
 * @brief Subsonic Outlet boundary condition with imposed static backpressure.
 */
class PressureOutletBC final : public IncompressibleBC {
public:
    PressureOutletBC(std::string zone,
                     const LocalIndex fbeg,
                     const LocalIndex fend,
                     const bc::physical::SubsonicOutletParams& p)
        : IncompressibleBC(std::move(zone), fbeg, fend), m_p(p) {}

    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const override {
        assert(q.size() == 4 && "PressureOutletBC requires variables [p, u, v, w]");
        bc::physical::kernels::subsonic_outlet_kernel<4>(q, mesh, this->m_begin, this->m_end, m_p);
    }

    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const override {
        assert(q.size() == 4 && gx.size() == 4 && gy.size() == 4 && gz.size() == 4);
        assert(q.size() == gx.size() && gx.size() == gy.size() && gy.size() == gz.size());

        bc::physical::kernels::subsonic_outlet_grad_kernel<4>(q, gx, gy, gz, mesh, this->m_begin, this->m_end, m_p);
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

    [[nodiscard]] bc::physical::BCType kind() const noexcept override { return bc::physical::BCType::SubsonicOutlet; }

private:
    bc::physical::SubsonicOutletParams m_p;
};

} // namespace cfd::solver::incompressible::physical_bc 
#pragma once

#include <span>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/bc/physical/physical_bc.hpp" // базовый BoundaryCondition

namespace cfd::solver::incompressible::physical_bc {

/**
 * @class IncompressibleBoundaryCondition
 * @brief Base class for incompressible Navier-Stokes boundaries.
 * Extends the generic ghost-filling interface with SLAE matrix assembly methods.
 */
class IncompressibleBoundaryCondition : public bc::physical::BoundaryCondition {
public:
    using cfd::bc::physical::BoundaryCondition::BoundaryCondition;

    /**
     * @brief Contributes boundary conditions to the momentum SLAE (velocity discretization).
     * @param[in,out] diag_u Diagonal coefficient a_P of the owner cell (for implicit scheme).
     * @param[in,out] rhs Right-hand side vector of the momentum equations [rhs_u, rhs_v, rhs_w].
     * @param[in] mesh Local mesh.
     */
    virtual void apply_momentum_bc(std::span<double*> diag_u,
                                   std::span<double*> rhs,
                                   const mesh::MeshPart& mesh) const = 0;

     /**
     * @brief Contributes boundary conditions to the Poisson equation for pressure correction p'.
     * @param[in,out] diag_p Diagonal coefficient of the Poisson matrix.
     * @param[in,out] rhs_p Mass balance residual vector at faces (Poisson right-hand side).
     * @param[in] mesh Local mesh.
     */
    virtual void apply_pressure_bc(std::span<double*> diag_p,
                                   std::span<double*> rhs_p,
                                   const mesh::MeshPart& mesh) const = 0;
};

using IncompressibleBC = IncompressibleBoundaryCondition;

} // namespace cfd::solver::incompressible::physical_bc
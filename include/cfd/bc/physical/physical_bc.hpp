#pragma once

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_geometry.hpp"

#include <string>
#include <span>

namespace cfd::bc::physical {

enum class BCType {
    // Physical (coupled) BCs:
    SupersonicInlet,
    SupersonicOutlet,
    SubsonicInlet,
    SubsonicOutlet,
    SlipWall,
    NoSlipWall,
    NoSlipWallHeatFlux,
    Symmetry,
    Farfield
};

enum class InflowMode {
    Velocity,      ///< Direct (u, v, w) components
    MachAngles,    ///< Mach number + alpha_deg + beta_deg
    MachDirection  ///< Mach number + unit direction vector [dx, dy, dz]
};

[[nodiscard]] constexpr const char* to_string(const BCType k) noexcept {
    switch (k) {
        case BCType::SupersonicInlet:      return "SUPERSONIC_INLET";
        case BCType::SupersonicOutlet:     return "SUPERSONIC_OUTLET";
        case BCType::SubsonicInlet:        return "SUBSONIC_INLET";
        case BCType::SubsonicOutlet:       return "SUBSONIC_OUTLET";
        case BCType::SlipWall:             return "SLIP_WALL";
        case BCType::NoSlipWall:           return "NO_SLIP_WALL";
        case BCType::NoSlipWallHeatFlux:   return "NO_SLIP_WALL_HEAT_FLUX";
        case BCType::Symmetry:             return "SYMMETRY";
        case BCType::Farfield:             return "FARFIELD";
    }
    return "UNKNOWN";
}

/**
 * @class BoundaryCondition
 * @brief Abstract strategy interface for physical boundary patches.
 * 
 * Ghost cells for patch faces start at index: c_ghost = n_cells + (face_idx - n_inner_faces).
 */
class BoundaryCondition {
public:
    /**
     * @param[in] zone - boundary condition zone (face_zone)
     * @param[in] fbeg, fbeg_loc - start of boundary patch face indices
     * @param[in] fend, fend_loc - end of boundary patch face indices
     * @note Indexation: [fbeg, fend)
     */
    BoundaryCondition(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : m_zone(std::move(zone)), m_begin(fbeg), m_end(fend) {}
    virtual ~BoundaryCondition() = default;

    // =========================
    // ==== GENERAL METHODS ====
    // =========================
    
    /**
     * @brief Patch apply subroutine
     * @param[inout] q - MeanFlow variables span [p u v w (T)]
     */
    virtual void update_ghost_cells(std::span<double* const> q, 
                                    const mesh::MeshPart& mesh) const = 0;

    /** 
     * @brief Patch apply gradients subroutine
     * @param[in] q - MeanFlow variables span [p u v w (T)]
     * @param[inout] gxyz - MeanFlow variables gradients spans [grad_pxyz grad_uxyz grad_vxyz grad_wxyz (grad_Txyz)]
     */
    virtual void update_ghost_cells_grad(std::span<const double* const> q, 
                                         std::span<double* const> gx, 
                                         std::span<double* const> gy, 
                                         std::span<double* const> gz, 
                                         const mesh::MeshPart& mesh) const = 0;
    
    // ================================
    // ==== INCOMPRESSIBLE METHODS ====
    // ================================

    /**
     * @brief Contributes boundary conditions to the momentum SLAE (velocity discretization).
     * @param[in,out] diag_u Diagonal coefficient a_P of the owner cell (for implicit scheme).
     * @param[in,out] rhs Right-hand side vector of the momentum equations [rhs_u, rhs_v, rhs_w].
     * @param[in] mesh Local mesh.
     */
    virtual void apply_momentum_bc(const mesh::MeshPart& mesh,
                                   const mesh::MeshAuxGeometry& aux_geom,
                                   double* CFD_RESTRICT diag,
                                   double* CFD_RESTRICT rhs_u,
                                   double* CFD_RESTRICT rhs_v,
                                   double* CFD_RESTRICT rhs_w,
                                   double* CFD_RESTRICT m_dot,
                                   const double rho,
                                   const double mu,
                                   const double* CFD_RESTRICT mut = nullptr) const = 0;
     /**
     * @brief Contributes boundary conditions to the Poisson equation for pressure correction p'.
     * @param[in,out] diag_p Diagonal coefficient of the Poisson matrix.
     * @param[in,out] rhs_p Mass balance residual vector at faces (Poisson right-hand side).
     * @param[in] mesh Local mesh.
     */
    virtual void apply_pressure_bc(std::span<double*> diag_p,
                                   std::span<double*> rhs_p,
                                   const mesh::MeshPart& mesh) const = 0;


    [[nodiscard]] virtual BCType kind() const noexcept = 0;
    
    [[nodiscard]] const std::string& zone() const noexcept { return m_zone; }
    [[nodiscard]] LocalIndex begin() const noexcept { return m_begin; }
    [[nodiscard]] LocalIndex end() const noexcept { return m_end; }
    [[nodiscard]] LocalIndex size() const noexcept { return m_end - m_begin; }

protected:
    std::string m_zone;
    LocalIndex m_begin{0};
    LocalIndex m_end{0};
};

} //namespace cfd::bc::physical

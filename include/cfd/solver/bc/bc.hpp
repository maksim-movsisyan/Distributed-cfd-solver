#pragma once

#include "cfd/core/types.hpp"
#include "cfd/solver/fields/fields_view.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/mesh/localmesh.hpp"
#include <string>

namespace cfd::solver::bc {

enum class BCType {
    SupersonicInlet,
    SupersonicOutlet,
    SlipWall,
    Symmetry,
    Farfield,
};

enum class InflowMode {
    Velocity,      ///< Direct (u, v, w) components
    MachAngles,    ///< Mach number + alpha_deg + beta_deg
    MachDirection  ///< Mach number + unit direction vector [dx, dy, dz]
};

[[nodiscard]] constexpr const char* to_string(const BCType k) noexcept {
    switch (k) {
        case BCType::SupersonicInlet:  return "SUPERSONIC_INLET";
        case BCType::SupersonicOutlet: return "SUPERSONIC_OUTLET";
        case BCType::SlipWall:         return "SLIP_WALL";
        case BCType::Symmetry:         return "SYMMETRY";
        case BCType::Farfield:         return "FARFIELD";
    }
    return "UNKNOWN";
}

/**
 * @class BoundaryCondition
 * @brief Abstract strategy interface for physical boundary patches.
 * 
 * Ghost cells for patch faces start at index: c_ghost = n_cells + (face_idx - n_inner_faces).
 */
template <eos::EquationOfState EOS>
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

    /** @brief Patch apply subroutine */
    virtual void apply(fields::PrimitiveView<double> state, 
                       const mesh::MeshPart& mesh,
                       const EOS& eos) const = 0;

    /** @brief Patch apply gradients subroutine */
    virtual void apply_grad(fields::PrimitiveView<const double> state, 
                            fields::PrimitiveGradView<double> state_grad, 
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

} //namespace cfd::solver::bc

#pragma once

#include <span>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::bc::scalar {

enum class ScalarBCType {
    Dirichlet,
    Neumann,
    Robin,
    Extrapolate
};

[[nodiscard]] constexpr const char* to_string(const ScalarBCType k) noexcept {
    switch (k) {
        case ScalarBCType::Dirichlet:   return "DIRICHLET";
        case ScalarBCType::Neumann:     return "NEUMANN";
        case ScalarBCType::Robin:       return "ROBIN";
        case ScalarBCType::Extrapolate: return "EXTRAPOLATE";
    }
    return "UNKNOWN";
}

/**
 * @class ScalarBoundaryCondition
 * @brief Abstract strategy interface for generic scalar field boundary conditions (turbulence, Poisson, species).
 */
class ScalarBoundaryCondition {
public:
    ScalarBoundaryCondition(std::string zone, const LocalIndex fbeg, const LocalIndex fend)
        : m_zone(std::move(zone)), m_begin(fbeg), m_end(fend) {}
    virtual ~ScalarBoundaryCondition() = default;

    /**
     * @brief Fills ghost cell scalar values.
     * @param[inout] phi Contiguous span of pointers to cell values [0, n_cells + n_ghost).
     * @param[in] mesh Local partitioned mesh.
     */
    virtual void update_ghost_cells(std::span<double *> phi,
                                    const mesh::MeshPart& mesh) const = 0;

    /**
     * @brief Fills ghost cell scalar gradients.
     * @param[in] phi Contiguous span of pointers to cell values [0, n_cells + n_ghost).
     * @param[in,out] gx Contiguous span of pointers to x-gradient values.
     * @param[in,out] gy Contiguous span of pointers to y-gradient values.
     * @param[in,out] gz Contiguous span of pointers to z-gradient values.
     * @param[in] mesh Local partitioned mesh.
     */
    virtual void update_ghost_cells_grad(std::span<const double *> phi,
                                         std::span<double *> gx,
                                         std::span<double *> gy,
                                         std::span<double *> gz,
                                         const mesh::MeshPart& mesh) const = 0;

    [[nodiscard]] virtual ScalarBCType kind() const noexcept = 0;

    [[nodiscard]] const std::string& zone() const noexcept { return m_zone; }
    [[nodiscard]] LocalIndex begin() const noexcept { return m_begin; }
    [[nodiscard]] LocalIndex end() const noexcept { return m_end; }
    [[nodiscard]] LocalIndex size() const noexcept { return m_end - m_begin; }

protected:
    std::string m_zone;
    LocalIndex m_begin{0};
    LocalIndex m_end{0};
};

using ScalarBC = ScalarBoundaryCondition;

} // namespace cfd::bc::scalar
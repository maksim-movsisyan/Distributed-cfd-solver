#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::bc {
/**
 * @class BoundaryManager
 * @brief Owns all boundary condition patch objects and dispatches ghost cell filling.
 */
class BoundaryManager {
public:
    using BCPtr = std::unique_ptr<BoundaryCondition>;
    using BCBuilder = std::function<BCPtr(const std::string& zone_label,
                                          LocalIndex beg, LocalIndex end,
                                          const BCDescriptor& bc_desc,
                                          double gamma, double R)>;

    BoundaryManager() { register_all(); }

    /**
     * @brief Builds and registers BC objects for all patches present on this MPI rank.
     * @param bc_config Parsed boundary configuration.
     * @param mesh Local partitioned mesh.
     * @param gamma Specific heat ratio [-].
     * @param gas_constant Specific gas constant R [J / (kg K)].
     */
    void initialize(const BoundaryConfig& bc_config,
                    const mesh::MeshPart& mesh,
                    double gamma,
                    double gas_constant);

    /**
     * @brief Fills ghost cell primitive states for all local boundary patches.
     */
    void apply_all(fields::PrimitiveView<double> state,
                   const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply(state, mesh);
        }
    }

    /**
     * @brief Fills ghost cell gradients for all local boundary patches.
     */
    void apply_grad_all(fields::PrimitiveView<const double> state,
                        fields::PrimitiveGradView<double> state_grad,
                        const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply_grad(state, state_grad, mesh);
        }
    }

    [[nodiscard]] std::size_t num_local_zones() const noexcept { return m_bcs.size(); }
    [[nodiscard]] const std::vector<BCPtr>& patches() const noexcept { return m_bcs; }
private:
    std::vector<BCPtr> m_bcs;
    std::unordered_map<BCType, BCBuilder> m_registry;   ///< bc_type -> bc builder

    /** @brief Registers all known BC builders (single place to extend). */
    void register_all();

    /**
     * @brief Scan host face_zone over the boundary range and build
     *        patch_id -> [begin, end) for each contiguous zone block.
     * @note  Requires boundary faces to be grouped by zone (mesh preprocessing).
     */
    std::unordered_map<LocalIndex, std::pair<LocalIndex, LocalIndex>> build_zone_ranges(const mesh::MeshPart& mesh) const;
};

} //namespace cfd::solver::bc
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/bc/physical/physical_bc.hpp"
#include "cfd/bc/physical/subsonic_inlet.hpp"
#include "cfd/bc/physical/subsonic_outlet.hpp"
#include "cfd/bc/physical/symmetry.hpp"
#include "cfd/bc/physical/noslip_wall.hpp"
#include "cfd/bc/config.hpp"

namespace cfd::solver::incompressible {

/**
 * @class BoundaryManager
 * @brief Owns all boundary condition patch objects and coordinates ghost cell filling.
 */
class BoundaryManager {
public:
    using BCPtr = std::unique_ptr<bc::physical::BoundaryCondition>;
    using BCBuilder = std::function<BCPtr(const std::string& zone_label,
                                          const LocalIndex beg,
                                          const LocalIndex end,
                                          const bc::BCDescriptor& bc_desc)>;

    BoundaryManager() { register_all(); }

    /**
     * @brief Builds and registers BC objects for all boundary patches present on this MPI rank.
     * 
     * @param bc_config Parsed boundary configuration.
     * @param mesh      Local partitioned mesh.
     */
    void initialize(const bc::BoundaryConfig& bc_config,
                    const mesh::MeshPart& mesh) {
        m_bcs.clear();

        // 1. Map patch_id to boundary face ranges: patch_id -> [start_face, end_face)
        const auto ranges = build_zone_ranges(mesh);

        // 2. Instantiate boundary conditions for local patches
        for (const auto& patch : bc_config.patches) {
            const auto it_range = ranges.find(static_cast<LocalIndex>(patch.patch_id));

            // Skip patch if this rank contains no boundary faces belonging to it
            if (it_range == ranges.end()) {
                continue;
            }

            const auto it_builder = m_registry.find(patch.type);
            if (it_builder == m_registry.end()) {
                throw std::runtime_error("BoundaryManager::initialize: unregistered BC type for patch " +
                                         std::to_string(patch.patch_id));
            }

            const std::string zone_name = std::string(to_string(patch.type)) + "-" + std::to_string(patch.patch_id);
            const auto [fbeg, fend] = it_range->second;

            if (fbeg < fend) {
                m_bcs.push_back(it_builder->second(zone_name, fbeg, fend, patch));
            }
        }
    }

    /**
     * @brief Fills ghost cell primitive states for all local boundary patches.
     * @param[in,out] q Span of SoA pointers to mean-flow primitive variables [p, u, v, w, T].
     * @param[in] mesh Local partitioned mesh.
     */
    void update_ghost_cells(std::span<double* const> q,
                            const mesh::MeshPart& mesh) const {
        assert(q.size() == 4 && "Incompressible BoundaryManager::update_ghost_cells requires exactly 4 mean-flow variables");
        for (const auto& bc : m_bcs) {
            bc->update_ghost_cells(q, mesh);
        }
    }

    /**
     * @brief Fills ghost cell gradients for all local boundary patches.
     * @param[in] q Span of SoA pointers to primitive variables.
     * @param[in,out] gx Span of SoA pointers to x-gradients.
     * @param[in,out] gy Span of SoA pointers to y-gradients.
     * @param[in,out] gz Span of SoA pointers to z-gradients.
     * @param[in] mesh Local partitioned mesh.
     */
    void update_ghost_cells_grad(std::span<const double* const> q,
                                 std::span<double* const> gx,
                                 std::span<double* const> gy,
                                 std::span<double* const> gz,
                                 const mesh::MeshPart& mesh) const {
        assert(q.size() == 4 && gx.size() == 4 && gy.size() == 4 && gz.size() == 4);
        for (const auto& bc : m_bcs) {
            bc->update_ghost_cells_grad(q, gx, gy, gz, mesh);
        }
    }

    /**
     * @brief Contributes boundary conditions to the momentum SLAE (velocity discretization).
     * @param[in,out] diag_u Diagonal coefficient a_P of the owner cell (for implicit scheme).
     * @param[in,out] rhs Right-hand side vector of the momentum equations [rhs_u, rhs_v, rhs_w].
     * @param[in] mesh Local mesh.
     */
    void apply_momentum_bc(std::span<double*> diag_u,
                           std::span<double*> rhs,
                           const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply_momentum_bc(diag_u, rhs, mesh);
        }
    }

     /**
     * @brief Contributes boundary conditions to the Poisson equation for pressure correction p'.
     * @param[in,out] diag_p Diagonal coefficient of the Poisson matrix.
     * @param[in,out] rhs_p Mass balance residual vector at faces (Poisson right-hand side).
     * @param[in] mesh Local mesh.
     */
    void apply_pressure_bc(std::span<double*> diag_p,
                           std::span<double*> rhs_p,
                           const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply_pressure_bc(diag_p, rhs_p, mesh);
        }
    }




    [[nodiscard]] std::size_t num_local_zones() const noexcept { return m_bcs.size(); }
    [[nodiscard]] const std::vector<BCPtr>& patches() const noexcept { return m_bcs; }

private:
    std::vector<BCPtr> m_bcs;
    std::unordered_map<bc::physical::BCType, BCBuilder> m_registry;

    /**
     * @brief Registers all known BC builders (Single Point of Extension).
     */
    void register_all() {
        // ==== Symmetry ====
        m_registry[bc::physical::BCType::Symmetry] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& /*desc*/) -> BCPtr {
                return std::make_unique<bc::physical::SymmetryBC>(z, b, e);
            };

        // ==== No-slip Wall (Isothermal) ====
        m_registry[bc::physical::BCType::NoSlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc) -> BCPtr {
                const bc::physical::NoSlipWallParams par{
                    .vx_wall  = desc.velocity[0],
                    .vy_wall  = desc.velocity[1],
                    .vz_wall  = desc.velocity[2],
                    .tmp_wall = desc.t
                };
                return std::make_unique<bc::physical::NoSlipWallBC>(z, b, e, par);
            };

        // ==== Subsonic Inlet ====
        m_registry[bc::physical::BCType::SubsonicInlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc) -> BCPtr {

                const bc::physical::SubsonicInletParams par{
                    .vx_inlet  = desc.velocity[0],
                    .vy_inlet  = desc.velocity[1],
                    .vz_inlet  = desc.velocity[2],
                    .tmp_inlet = desc.t
                };
                return std::make_unique<bc::physical::SubsonicInletBC>(z, b, e, par);
            };

        // ==== Subsonic Outlet ====
        m_registry[bc::physical::BCType::SubsonicOutlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc) -> BCPtr {
                const bc::physical::SubsonicOutletParams par{
                    .prs_outlet = desc.p
                };
                return std::make_unique<bc::physical::SubsonicOutletBC>(z, b, e, par);
            };

        // ==== Slip wall ====
        m_registry[bc::physical::BCType::SlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
            const bc::BCDescriptor& /*desc*/) -> BCPtr {
                return std::make_unique<bc::physical::SymmetryBC>(z, b, e);
            };
    }

    /**
     * @brief Scan face_patch over the boundary range [n_inner_faces, n_faces)
     *        and build patch_id -> [begin, end) ranges.
     */
    std::unordered_map<LocalIndex, std::pair<LocalIndex, LocalIndex>>
    build_zone_ranges(const mesh::MeshPart& mesh) const {
        const auto& face_patch = mesh.face_patch;
        const auto n_faces     = static_cast<LocalIndex>(mesh.n_faces);
        const auto bf_begin    = static_cast<LocalIndex>(mesh.n_inner_faces);

        std::unordered_map<LocalIndex, std::pair<LocalIndex, LocalIndex>> ranges;

        LocalIndex f = bf_begin;
        while (f < n_faces) {
            const auto zid = static_cast<LocalIndex>(face_patch[static_cast<std::size_t>(f)]);

            if (zid < 0) {
                mpi::fatal(MPI_COMM_WORLD, "BoundaryManager::build_zone_ranges: negative patch id detected");
            }

            const LocalIndex start = f;
            while (f < n_faces && static_cast<LocalIndex>(face_patch[static_cast<std::size_t>(f)]) == zid) {
                ++f;
            }
            ranges[zid] = {start, f};
        }
        return ranges;
    }

};

} //namespace cfd::solver::incompressible
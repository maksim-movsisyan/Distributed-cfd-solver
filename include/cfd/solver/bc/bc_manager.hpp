#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/bc/bc_farfield.hpp"
#include "cfd/solver/bc/bc_slip_wall.hpp"
#include "cfd/solver/bc/bc_supersonic_inlet.hpp"
#include "cfd/solver/bc/bc_supersonic_outlet.hpp"
#include "cfd/solver/bc/bc_symmetry.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/eos/eos_concept.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::bc {
/**
 * @class BoundaryManager
 * @brief Owns all boundary condition patch objects and coordinates ghost cell filling.
 * 
 * @tparam EOS Thermodynamic Equation of State conforming to eos::EquationOfState.
 */
 template <eos::EquationOfState EOS>
class BoundaryManager {
public:
    using BCPtr = std::unique_ptr<BoundaryCondition<EOS>>;
    using BCBuilder = std::function<BCPtr(const std::string& zone_label,
                                          const LocalIndex beg,
                                          const LocalIndex end,
                                          const BCDescriptor& bc_desc,
                                          const EOS& eos)>;

    BoundaryManager() { register_all(); }

    /**
     * @brief Builds and registers BC objects for all boundary patches present on this MPI rank.
     * 
     * @param bc_config Parsed boundary configuration.
     * @param mesh      Local partitioned mesh.
     * @param eos       Thermodynamic equation of state.
     */
    void initialize(const BoundaryConfig& bc_config,
                    const mesh::MeshPart& mesh,
                    const EOS& eos) {
        m_eos = eos;
        m_bcs.clear();

        // 1. Create zone ranges: patch_id -> [start_face, end_face)
        const auto ranges = build_zone_ranges(mesh);

        // 2. Instantiate boundary conditions for local patches
        for (const auto& patch : bc_config.patches) {
            const auto it_range = ranges.find(static_cast<LocalIndex>(patch.patch_id));

            // If this MPI rank has no boundary faces belonging to this patch, skip cleanly
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
                m_bcs.push_back(it_builder->second(zone_name, fbeg, fend, patch, m_eos));
            }
        }
    }

    /**
     * @brief Fills ghost cell primitive states for all local boundary patches.
     */
    void apply_all(fields::PrimitiveView<double> state,
                   const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply(state, mesh, m_eos);
        }
    }

    /**
     * @brief Fills ghost cell gradients for all local boundary patches (2nd-order reconstruction).
     */
    void apply_grad_all(fields::ConstPrimitiveView state,
                        fields::PrimitiveGradView<double> state_grad,
                        const mesh::MeshPart& mesh) const {
        for (const auto& bc : m_bcs) {
            bc->apply_grad(state, state_grad, mesh);
        }
    }

    [[nodiscard]] std::size_t num_local_zones() const noexcept { return m_bcs.size(); }
    [[nodiscard]] const std::vector<BCPtr>& patches() const noexcept { return m_bcs; }

private:
    EOS m_eos{};
    std::vector<BCPtr> m_bcs;
    std::unordered_map<BCType, BCBuilder> m_registry;

    /**
     * @brief Registers all known BC builders (Single Point of Extension).
     */
    void register_all() {
        // ==== Supersonic Inlet ====
        m_registry[BCType::SupersonicInlet] =
        [](const std::string& z, const LocalIndex b, const LocalIndex e,
           const BCDescriptor& desc, const EOS& eos) -> BCPtr {
            SupersonicInletParams par;

            switch (desc.inflow_mode) {
                case InflowMode::Velocity:
                    par = SupersonicInletParams::from_velocities(
                        desc.p,
                        desc.velocity[0],
                        desc.velocity[1],
                        desc.velocity[2],
                        desc.t
                    );
                    break;

                case InflowMode::MachAngles:
                    par = SupersonicInletParams::from_mach_angles(
                        eos,
                        desc.p,
                        desc.t,
                        desc.mach,
                        desc.alpha_deg,
                        desc.beta_deg
                    );
                    break;

                case InflowMode::MachDirection:
                    par = SupersonicInletParams::from_mach_direction(
                        eos,
                        desc.p,
                        desc.t,
                        desc.mach,
                        desc.direction[0],
                        desc.direction[1],
                        desc.direction[2]
                    );
                    break;
            }

            return std::make_unique<SupersonicInletBC<EOS>>(z, b, e, par);
        };

        // ==== Supersonic Outlet ====
        m_registry[BCType::SupersonicOutlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<SupersonicOutletBC<EOS>>(z, b, e);
            };

        // ==== Slip Wall ====
        m_registry[BCType::SlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<SlipWallBC<EOS>>(z, b, e);
            };

        // ==== Symmetry ====
        m_registry[BCType::Symmetry] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<SymmetryBC<EOS>>(z, b, e);
            };

        // ==== Farfield (Riemann Invariants) ====
        m_registry[BCType::Farfield] =
        [](const std::string& z, const LocalIndex b, const LocalIndex e,
           const BCDescriptor& desc, const EOS& eos) -> BCPtr {
            FarfieldParams par;

            switch (desc.inflow_mode) {
                case InflowMode::Velocity:
                    par = FarfieldParams::from_velocities(
                        eos,
                        desc.p,
                        desc.velocity[0],
                        desc.velocity[1],
                        desc.velocity[2],
                        desc.t
                    );
                    break;

                case InflowMode::MachAngles:
                    par = FarfieldParams::from_mach_angles(
                        eos,
                        desc.p,
                        desc.t,
                        desc.mach,
                        desc.alpha_deg,
                        desc.beta_deg
                    );
                    break;

                case InflowMode::MachDirection:
                    par = FarfieldParams::from_mach_direction(
                        eos,
                        desc.p,
                        desc.t,
                        desc.mach,
                        desc.direction[0],
                        desc.direction[1],
                        desc.direction[2]
                    );
                    break;
            }

            return std::make_unique<FarfieldBC<EOS>>(z, b, e, par);
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

} //namespace cfd::solver::bc
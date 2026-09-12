#pragma once

#include <cassert>
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
#include "cfd/solver/bc/bc.hpp"
#include "cfd/solver/bc/config.hpp"
#include "cfd/solver/bc/farfield.hpp"
#include "cfd/solver/bc/noslip_wall.hpp"
#include "cfd/solver/bc/noslip_wall_heat_flux.hpp"
#include "cfd/solver/bc/slip_wall.hpp"
#include "cfd/solver/bc/subsonic_inlet.hpp"
#include "cfd/solver/bc/subsonic_outlet.hpp"
#include "cfd/solver/bc/supersonic_inlet.hpp"
#include "cfd/solver/bc/supersonic_outlet.hpp"
#include "cfd/solver/bc/symmetry.hpp"
#include "cfd/solver/eos/concepts.hpp"

namespace cfd::solver::bc {

/**
 * @class BoundaryManager
 * @brief Owns all boundary condition patch objects and coordinates ghost cell filling.
 * 
 * @tparam EOS Thermodynamic Equation of State conforming to physics::eos::EquationOfStatePolicy.
 */
template <eos::EquationOfStatePolicy EOS>
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
     * @param eos       Thermodynamic equation of state model.
     */
    void initialize(const BoundaryConfig& bc_config,
                    const mesh::MeshPart& mesh,
                    const EOS& eos) {
        m_eos = eos;
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
                m_bcs.push_back(it_builder->second(zone_name, fbeg, fend, patch, m_eos));
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
        assert(q.size() == 5 && "BoundaryManager::update_ghost_cells requires exactly 5 mean-flow variables");
        for (const auto& bc : m_bcs) {
            bc->update_ghost_cells(q, mesh, m_eos);
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
        assert(q.size() == 5 && gx.size() == 5 && gy.size() == 5 && gz.size() == 5);
        for (const auto& bc : m_bcs) {
            bc->update_ghost_cells_grad(q, gx, gy, gz, mesh);
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

        // ==== No-slip Wall (Isothermal) ====
        m_registry[BCType::NoSlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const auto par = NoSlipWallParams::moving_isothermal(
                    desc.velocity[0],
                    desc.velocity[1],
                    desc.velocity[2],
                    desc.t
                );
                return std::make_unique<NoSlipWallBC<EOS>>(z, b, e, par);
            };

        // ==== No-slip Wall (Fixed Temperature Gradient / Heat Flux) ====
        m_registry[BCType::NoSlipWallHeatFlux] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const auto par = NoSlipWallHeatFluxParams::moving_gradient(
                    desc.velocity[0],
                    desc.velocity[1],
                    desc.velocity[2],
                    desc.tmp_grad
                );
                return std::make_unique<NoSlipWallHeatFluxBC<EOS>>(z, b, e, par);
            };
        
        // ==== Subsonic Inlet ====
        m_registry[BCType::SubsonicInlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& desc, const EOS& eos) -> BCPtr {
                SubsonicInletParams par;

                switch (desc.inflow_mode) {
                    case InflowMode::Velocity:
                        par = SubsonicInletParams::from_velocities(
                            desc.velocity[0],
                            desc.velocity[1],
                            desc.velocity[2],
                            desc.t
                        );
                        break;

                    case InflowMode::MachAngles:
                        par = SubsonicInletParams::from_mach_angles(
                            eos,
                            desc.t,
                            desc.mach,
                            desc.alpha_deg,
                            desc.beta_deg
                        );
                        break;

                    case InflowMode::MachDirection:
                        par = SubsonicInletParams::from_mach_direction(
                            eos,
                            desc.t,
                            desc.mach,
                            desc.direction[0],
                            desc.direction[1],
                            desc.direction[2]
                        );
                        break;
                }

                return std::make_unique<SubsonicInletBC<EOS>>(z, b, e, par);
            };

        // ==== Subsonic Outlet ====
        m_registry[BCType::SubsonicOutlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const auto par = SubsonicOutletParams::from_pressure(desc.p);
                return std::make_unique<SubsonicOutletBC<EOS>>(z, b, e, par);
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
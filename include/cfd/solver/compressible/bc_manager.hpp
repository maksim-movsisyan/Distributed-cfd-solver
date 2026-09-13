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
#include "cfd/bc/physical/farfield.hpp"
#include "cfd/bc/physical/noslip_wall.hpp"
#include "cfd/bc/physical/noslip_wall_heat_flux.hpp"
#include "cfd/bc/physical/slip_wall.hpp"
#include "cfd/bc/physical/subsonic_inlet.hpp"
#include "cfd/bc/physical/subsonic_outlet.hpp"
#include "cfd/bc/physical/supersonic_inlet.hpp"
#include "cfd/bc/physical/supersonic_outlet.hpp"
#include "cfd/bc/physical/symmetry.hpp"
#include "cfd/bc/config.hpp"
#include "cfd/solver/compressible/eos/concepts.hpp"

namespace cfd::solver::compressible {

namespace {

struct InflowPrimitiveState {
    double p{101325.0};
    double u{0.0};
    double v{0.0};
    double w{0.0};
    double T{288.15};
};

// Direct primitive values
inline InflowPrimitiveState from_velocities(const double p_in, 
                                           const double u_in, 
                                           const double v_in, 
                                           const double w_in, 
                                           const double T_in) noexcept {
    return {p_in, u_in, v_in, w_in, T_in};
}

// Mach number + unit direction vector (dx, dy, dz)
template <eos::EquationOfStatePolicy EOS>
inline InflowPrimitiveState from_mach_direction(const EOS& eos,
                                                const double p_in, 
                                                const double T_in, 
                                                const double mach,
                                                const double dx, 
                                                const double dy, 
                                                const double dz) noexcept {
    const double rho   = eos.density_Tp(T_in, p_in);
    const double a     = eos.sound_speed_rhop(rho, p_in);
    const double v_mag = mach * a;

    const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double inv_norm = (norm > 1.0e-14) ? (1.0 / norm) : 0.0;

    return {
        p_in,
        v_mag * dx * inv_norm,
        v_mag * dy * inv_norm,
        v_mag * dz * inv_norm,
        T_in
    };
}

// Mach number + aerodynamic angles (alpha, beta in degrees)
template <eos::EquationOfStatePolicy EOS>
inline InflowPrimitiveState from_mach_angles(const EOS& eos,
                                             const double p_in, 
                                             const double T_in, 
                                             const double mach,
                                             const double alpha_deg, 
                                             const double beta_deg) noexcept {
    constexpr double kDegToRad = M_PI / 180.0;
    const double alpha_rad = alpha_deg * kDegToRad;
    const double beta_rad  = beta_deg  * kDegToRad;

    const double rho   = eos.density_Tp(T_in, p_in);
    const double a     = eos.sound_speed_rhop(rho, p_in);
    const double v_mag = mach * a;

    return {
        p_in,
        v_mag * std::cos(alpha_rad) * std::cos(beta_rad),
        -v_mag * std::sin(beta_rad),
        v_mag * std::sin(alpha_rad) * std::cos(beta_rad),
        T_in
    };
}

template <eos::EquationOfStatePolicy EOS>
inline InflowPrimitiveState resolve_inflow(const bc::BCDescriptor& desc, 
                                           const EOS& eos, 
                                           const double p_ref) noexcept {
    switch (desc.inflow_mode) {
        case bc::physical::InflowMode::Velocity:
            return from_velocities(p_ref, desc.velocity[0], desc.velocity[1], desc.velocity[2], desc.t);

        case bc::physical::InflowMode::MachAngles:
            return from_mach_angles(eos, p_ref, desc.t, desc.mach, desc.alpha_deg, desc.beta_deg);

        case bc::physical::InflowMode::MachDirection:
            return from_mach_direction(eos, p_ref, desc.t, desc.mach, desc.direction[0], desc.direction[1], desc.direction[2]);
    }
    return {};
}

} // namespace 


/**
 * @class BoundaryManager
 * @brief Owns all boundary condition patch objects and coordinates ghost cell filling.
 * 
 * @tparam EOS Thermodynamic Equation of State conforming to physics::eos::EquationOfStatePolicy.
 */
template <eos::EquationOfStatePolicy EOS>
class BoundaryManager {
public:
    using BCPtr = std::unique_ptr<bc::physical::BoundaryCondition>;
    using BCBuilder = std::function<BCPtr(const std::string& zone_label,
                                          const LocalIndex beg,
                                          const LocalIndex end,
                                          const bc::BCDescriptor& bc_desc,
                                          const EOS& eos)>;

    BoundaryManager() { register_all(); }

    /**
     * @brief Builds and registers BC objects for all boundary patches present on this MPI rank.
     * 
     * @param bc_config Parsed boundary configuration.
     * @param mesh      Local partitioned mesh.
     * @param eos       Thermodynamic equation of state model.
     */
    void initialize(const bc::BoundaryConfig& bc_config,
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
        assert(q.size() == 5 && "Compressible BoundaryManager::update_ghost_cells requires exactly 5 mean-flow variables");
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
    std::unordered_map<bc::physical::BCType, BCBuilder> m_registry;

    /**
     * @brief Registers all known BC builders (Single Point of Extension).
     */
    void register_all() {
        // ==== Supersonic Inlet ====
        m_registry[bc::physical::BCType::SupersonicInlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& eos) -> BCPtr {
                const auto s = resolve_inflow(desc, eos, desc.p);
                const bc::physical::SupersonicInletParams par{
                    .prs_inlet = s.p,
                    .vx_inlet  = s.u,
                    .vy_inlet  = s.v,
                    .vz_inlet  = s.w,
                    .tmp_inlet = s.T
                };
                return std::make_unique<bc::physical::SupersonicInletBC>(z, b, e, par);
            };

        // ==== Supersonic Outlet ====
        m_registry[bc::physical::BCType::SupersonicOutlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<bc::physical::SupersonicOutletBC>(z, b, e);
            };

        // ==== Slip Wall ====
        m_registry[bc::physical::BCType::SlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<bc::physical::SlipWallBC>(z, b, e);
            };

        // ==== Symmetry ====
        m_registry[bc::physical::BCType::Symmetry] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& /*desc*/, const EOS& /*eos*/) -> BCPtr {
                return std::make_unique<bc::physical::SymmetryBC>(z, b, e);
            };

        // ==== Farfield (Riemann Invariants) ====
        m_registry[bc::physical::BCType::Farfield] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& eos) -> BCPtr {
                const auto s = resolve_inflow(desc, eos, desc.p);
                const bc::physical::FarfieldParams par{
                    .prs_inf = s.p,
                    .tmp_inf = s.T,
                    .vx_inf  = s.u,
                    .vy_inf  = s.v,
                    .vz_inf  = s.w,
                    .gamma   = eos.gamma(),
                    .R       = eos.gas_constant()
                };
                return std::make_unique<bc::physical::FarfieldBC>(z, b, e, par);
            };

        // ==== No-slip Wall (Isothermal) ====
        m_registry[bc::physical::BCType::NoSlipWall] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const bc::physical::NoSlipWallParams par{
                    .vx_wall  = desc.velocity[0],
                    .vy_wall  = desc.velocity[1],
                    .vz_wall  = desc.velocity[2],
                    .tmp_wall = desc.t
                };
                return std::make_unique<bc::physical::NoSlipWallBC>(z, b, e, par);
            };

        // ==== No-slip Wall (Fixed Temperature Gradient / Heat Flux) ====
        m_registry[bc::physical::BCType::NoSlipWallHeatFlux] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const bc::physical::NoSlipWallHeatFluxParams par{
                    .vx_wall       = desc.velocity[0],
                    .vy_wall       = desc.velocity[1],
                    .vz_wall       = desc.velocity[2],
                    .tmp_grad_wall = desc.tmp_grad
                };
                return std::make_unique<bc::physical::NoSlipWallHeatFluxBC>(z, b, e, par);
            };
        
        // ==== Subsonic Inlet ====
        m_registry[bc::physical::BCType::SubsonicInlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& eos) -> BCPtr {
                constexpr double kRefPressure = 101325.0;
                const double p_ref = (desc.p > 0.0) ? desc.p : kRefPressure;
                const auto s = resolve_inflow(desc, eos, p_ref);

                const bc::physical::SubsonicInletParams par{
                    .vx_inlet  = s.u,
                    .vy_inlet  = s.v,
                    .vz_inlet  = s.w,
                    .tmp_inlet = s.T
                };
                return std::make_unique<bc::physical::SubsonicInletBC>(z, b, e, par);
            };

        // ==== Subsonic Outlet ====
        m_registry[bc::physical::BCType::SubsonicOutlet] =
            [](const std::string& z, const LocalIndex b, const LocalIndex e,
               const bc::BCDescriptor& desc, const EOS& /*eos*/) -> BCPtr {
                const bc::physical::SubsonicOutletParams par{
                    .prs_outlet = desc.p
                };
                return std::make_unique<bc::physical::SubsonicOutletBC>(z, b, e, par);
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

} //namespace cfd::solver::compressible
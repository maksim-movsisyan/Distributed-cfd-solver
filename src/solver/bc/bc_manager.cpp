#include "cfd/solver/bc/bc_manager.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/solver/bc/bc_farfield.hpp"
#include "cfd/solver/bc/bc_slip_wall.hpp"
#include "cfd/solver/bc/bc_supersonic_inlet.hpp"
#include "cfd/solver/bc/bc_supersonic_outlet.hpp"
#include "cfd/solver/bc/bc_symmetry.hpp"
#include "cfd/solver/config.hpp"

namespace cfd::solver::bc {

//================================  
// ==== BC builders registry ====
//================================ 
void BoundaryManager::register_all() {
    // ==== supersonic inlet1: p, v, T ====
    m_registry[BCType::SupersonicInlet] =
        [](const std::string& z, LocalIndex b, LocalIndex e,
           const BCDescriptor& bc_desc, double /*gamma*/, double /*R*/) -> BCPtr {
            SupersonicInletParams par;
            par.prs_inlet = bc_desc.p;
            par.tmp_inlet = bc_desc.t;
            par.vx_inlet  = bc_desc.velocity[0];
            par.vy_inlet  = bc_desc.velocity[1];
            par.vz_inlet  = bc_desc.velocity[2];
            return std::make_unique<SupersonicInletBC>(z, b, e, par);
        };

    // ==== supersonic outlet: no parameters ====
    m_registry[BCType::SupersonicOutlet] =
        [](const std::string& z, LocalIndex b, LocalIndex e,
           const BCDescriptor& /*bc_desc*/, double /*gamma*/, double /*R*/) -> BCPtr {
            return std::make_unique<SupersonicOutletBC>(z, b, e);
        };
    
    // ==== far field: p, v, T ====
    m_registry[BCType::Farfield] =
        [](const std::string& z, LocalIndex b, LocalIndex e,
           const BCDescriptor& bc_desc, double gamma, double R) -> BCPtr {
            FarfieldParams par;
            par.prs_inf = bc_desc.p_inf;
            par.tmp_inf = bc_desc.t_inf;
            par.vx_inf  = bc_desc.velocity_inf[0];
            par.vy_inf  = bc_desc.velocity_inf[1];
            par.vz_inf  = bc_desc.velocity_inf[2];
            par.gamma   = gamma;
            par.R       = R;
            return std::make_unique<FarfieldBC>(z, b, e, par);
        };

    // ==== wall slip: no parameters ====
    m_registry[BCType::SlipWall] =
        [](const std::string& z, LocalIndex b, LocalIndex e,
           const BCDescriptor& /*bc_desc*/, double /*gamma*/, double /*R*/) -> BCPtr {
            return std::make_unique<SlipWallBC>(z, b, e);
        };
    // ==== symmetry: no parameters ====
    m_registry[BCType::Symmetry] =
        [](const std::string& z, LocalIndex b, LocalIndex e,
           const BCDescriptor& /*bc_desc*/, double /*gamma*/, double /*R*/) -> BCPtr {
            return std::make_unique<SymmetryBC>(z, b, e);
        };
}

//=======================================================
// ==== zone_id -> [begin, end) from host face_zone ====
//=======================================================
std::unordered_map<LocalIndex, std::pair<LocalIndex, LocalIndex>>
BoundaryManager::build_zone_ranges(const mesh::MeshPart& mesh) const {
    const auto& face_patch = mesh.face_patch;
    const LocalIndex n_faces = mesh.n_faces;
    const LocalIndex bf_begin = mesh.n_inner_faces;

    std::unordered_map<LocalIndex, std::pair<LocalIndex, LocalIndex>> ranges;

    LocalIndex f = bf_begin;
    while (f < n_faces) {
        const auto zid = static_cast<LocalIndex>(face_patch[static_cast<std::size_t>(f)]);

        if (zid < 0) {
            mpi::fatal(MPI_COMM_WORLD, "BoundaryManager::build_zone_ranges: Error in faces ordering (negative patch id)");
        }

        const LocalIndex start = f;
        while (f < n_faces && face_patch[static_cast<std::size_t>(f)] == zid) {
            ++f;
        }
        ranges[zid] = {start, f};
    }
    return ranges;
}

//=========================== 
// ==== BC construction ====
//===========================
void BoundaryManager::initialize(const BoundaryConfig& bc_config,
                                 const mesh::MeshPart& mesh,
                                 const double gamma,
                                 const double gas_constant) {
    m_bcs.clear();

    // ==== Create zone ranges: zone-id -> [start, end) ====
    const auto ranges = build_zone_ranges(mesh);

    // ==== Create boundary condition vector ====
    for (const auto& patch : bc_config.patches) {
        const auto it_range = ranges.find(static_cast<LocalIndex>(patch.patch_id));

        // If this rank has no boundary faces belonging to this patch, skip cleanly
        if (it_range == ranges.end()) {
            continue;
        }

        const auto it_builder = m_registry.find(patch.type);
        if (it_builder == m_registry.end()) {
            throw std::runtime_error("BoundaryManager::initialize: unregistered BC type for patch " +
                                     std::to_string(patch.patch_id));
        }

        std::string zone_name = std::string(to_string(patch.type)) + "-" + std::to_string(patch.patch_id);
        const auto [fbeg, fend] = it_range->second;

        if (fbeg < fend) {
            m_bcs.push_back(it_builder->second(zone_name, fbeg, fend, patch, gamma, gas_constant));
        }
    }

}


} //namespace cfd::solver::bc
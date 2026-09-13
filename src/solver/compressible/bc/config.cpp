#include "cfd/solver/compressible/bc/config.hpp"

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "cfd/io/toml/toml_utilities.hpp"
#include "cfd/mpi/log.hpp"

using namespace cfd::io::toml_utils;

namespace cfd::solver::compressible::bc {

namespace {
[[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

/**
 * @brief Helper parsing and strictly validating Inflow / Farfield parameter sets.
 */
void parse_inflow_descriptor(const toml::table& t,
                             BCDescriptor& d,
                             const std::string& ctx,
                             const MPI_Comm comm) {
    const bool has_vel = t.contains("velocity") || t.contains("velocity_inf");
    const bool has_mach = t.contains("mach") || t.contains("mach_inf");
    const bool has_dir = t.contains("direction");
    const bool has_angles = t.contains("alpha") || t.contains("alpha_deg") ||
                            t.contains("beta")  || t.contains("beta_deg");

    // 1. Conflict checks (Mutual Exclusivity)
    if (has_vel && (has_mach || has_dir || has_angles)) {
        fail(comm, ctx + ": conflicting parameters: cannot specify direct 'velocity' "
                         "together with 'mach', 'direction', or 'alpha/beta' angles");
    }
    if (has_dir && has_angles) {
        fail(comm, ctx + ": conflicting parameters: cannot specify both 'direction' vector "
                         "and 'alpha/beta' angles");
    }
    if (has_dir && !has_mach) {
        fail(comm, ctx + ": 'direction' vector requires 'mach' to be specified");
    }
    if (has_angles && !has_mach) {
        fail(comm, ctx + ": 'alpha/beta' angles require 'mach' to be specified");
    }
    if (!has_vel && !has_mach) {
        fail(comm, ctx + ": missing velocity definition: specify either 'velocity' vector "
                         "or 'mach' with angles/direction");
    }

    // 2. Strict key whitelisting per detected mode
    if (has_vel) {
        d.inflow_mode = bc::InflowMode::Velocity;
        check_allowed_keys(t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                               "p", "p_inf", "t", "t_inf", "velocity", "velocity_inf"},
                           ctx, comm);
    } else if (has_dir) {
        d.inflow_mode = bc::InflowMode::MachDirection;
        check_allowed_keys(t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                               "p", "p_inf", "t", "t_inf", "mach", "", "direction"},
                           ctx, comm);
    } else {
        d.inflow_mode = bc::InflowMode::MachAngles;
        check_allowed_keys(t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                               "p", "p_inf", "t", "t_inf", "mach",
                               "alpha", "alpha_deg", "beta", "beta_deg"},
                           ctx, comm);
    }

    // 3. Pressure & Temperature
    if (t.contains("p")) {
        d.p = req_number(t, "p", ctx, comm);
    } else if (t.contains("p_inf")) {
        d.p = req_number(t, "p_inf", ctx, comm);
    } else {
        fail(comm, ctx + ": missing required pressure ('p' or 'p_inf')");
    }

    if (t.contains("t")) {
        d.t = req_number(t, "t", ctx, comm);
    } else if (t.contains("t_inf")) {
        d.t = req_number(t, "t_inf", ctx, comm);
    } else {
        fail(comm, ctx + ": missing required temperature ('t' or 't_inf')");
    }

    check_positive(d.p, "p", ctx, comm);
    check_positive(d.t, "t", ctx, comm);

    // 4. Mode-specific payload extraction
    switch (d.inflow_mode) {
        case bc::InflowMode::Velocity: {
            const std::string vkey = t.contains("velocity") ? "velocity" : "velocity_inf";
            d.velocity = req_vec3(t, vkey.c_str(), ctx, comm);
            break;
        }

        case bc::InflowMode::MachDirection: {
            d.mach = req_number(t, "mach", ctx, comm);
            check_positive(d.mach, "mach", ctx, comm);
            d.direction = req_vec3(t, "direction", ctx, comm);

            const double mag2 = d.direction[0] * d.direction[0] +
                                d.direction[1] * d.direction[1] +
                                d.direction[2] * d.direction[2];
            if (mag2 < 1.0e-14) {
                fail(comm, ctx + ": 'direction' vector cannot be zero");
            }
            break;
        }

        case bc::InflowMode::MachAngles: {
            d.mach = req_number(t, "mach", ctx, comm);
            check_positive(d.mach, "mach", ctx, comm);

            if (t.contains("alpha_deg")) {
                d.alpha_deg = req_number(t, "alpha_deg", ctx, comm);
            } else if (t.contains("alpha")) {
                d.alpha_deg = req_number(t, "alpha", ctx, comm);
            } else {
                d.alpha_deg = 0.0;
            }

            if (t.contains("beta_deg")) {
                d.beta_deg = req_number(t, "beta_deg", ctx, comm);
            } else if (t.contains("beta")) {
                d.beta_deg = req_number(t, "beta", ctx, comm);
            } else {
                d.beta_deg = 0.0;
            }
            break;
        }
    }
}

} // anonymous namespace

BoundaryConfig parse_boundary_config(const std::string& path,
                                     const mesh::MeshPart& mp,
                                     const MPI_Comm comm) {
    const std::string raw_content = broadcast_file_content(path, comm);
    const toml::table root = parse_in_memory_or_die(raw_content, path, comm);

    check_allowed_keys(root, {"boundary_condition"}, "'" + path + "'", comm);

    const toml::array* arr = root.get_as<toml::array>("boundary_condition");
    if (arr == nullptr || arr->empty()) {
        fail(comm, "'" + path + "': no [[boundary_condition]] entries");
    }

    const std::size_t n_patches = mp.patches.size();
    BoundaryConfig out;
    out.patches.resize(n_patches);
    std::vector<char> seen(n_patches, 0);

    const std::initializer_list<const char*> meta = {
        "patch_id", "type", "name", "cgns_type", "global_face_count"
    };

    std::size_t idx = 0;
    for (const auto& item : *arr) {
        const toml::table* t = item.as_table();
        const std::string ctx = "'" + path + "' boundary_condition[" + std::to_string(idx) + "]";
        if (t == nullptr) {
            fail(comm, ctx + " must be a TOML table");
        }

        const std::int64_t pid = req_integer(*t, "patch_id", ctx, comm);
        if (pid < 0 || pid >= static_cast<std::int64_t>(n_patches)) {
            fail(comm, ctx + ": patch_id " + std::to_string(pid) +
                           " outside mesh range [0, " + std::to_string(n_patches) + ")");
        }
        if (seen[static_cast<std::size_t>(pid)] != 0) {
            fail(comm, ctx + ": duplicate condition for patch " + std::to_string(pid));
        }
        seen[static_cast<std::size_t>(pid)] = 1;

        BCDescriptor& d = out.patches[static_cast<std::size_t>(pid)];
        d.patch_id = static_cast<int>(pid);
        const std::string type = req_string(*t, "type", ctx, comm);

        if (type == "SUPERSONIC_INLET") {
            d.type = bc::BCType::SupersonicInlet;
            parse_inflow_descriptor(*t, d, ctx, comm);

        } else if (type == "FARFIELD") {
            d.type = bc::BCType::Farfield;
            parse_inflow_descriptor(*t, d, ctx, comm);

            if (d.inflow_mode == bc::InflowMode::MachAngles || d.inflow_mode == bc::InflowMode::MachDirection) {
                mpi::log_stat("WARNING: %s: Mach-based inflow mode selected. "
                                "Note: velocity magnitude is evaluated assuming Ideal Gas EOS kinematics.",
                                ctx.c_str());
            }
        } else if (type == "SUPERSONIC_OUTLET") {
            d.type = bc::BCType::SupersonicOutlet;
            check_allowed_keys(*t, meta, ctx, comm);

        } else if (type == "SLIP_WALL") {
            d.type = bc::BCType::SlipWall;
            check_allowed_keys(*t, meta, ctx, comm);

        } else if (type == "SYMMETRY") {
            d.type = bc::BCType::Symmetry;
            check_allowed_keys(*t, meta, ctx, comm);
        } else if (type == "NO_SLIP_WALL") {
            d.type = bc::BCType::NoSlipWall;
            check_allowed_keys(*t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                                    "t", "t_wall", "velocity"}, ctx, comm);

            if (t->contains("t_wall")) {
                d.t = req_number(*t, "t_wall", ctx, comm);
            } else if (t->contains("t")) {
                d.t = req_number(*t, "t", ctx, comm);
            } else {
                d.t = constants::kIsaTemperature; // default isothermal wall temperature
            }
            check_positive(d.t, "t", ctx, comm);

            if (t->contains("velocity")) {
                d.velocity = req_vec3(*t, "velocity", ctx, comm);
            } else {
                d.velocity = {0.0, 0.0, 0.0}; // default stationary wall
            }
        } else if (type == "NO_SLIP_WALL_HEAT_FLUX" || type == "NO_SLIP_WALL_ADIABATIC") {
            d.type = bc::BCType::NoSlipWallHeatFlux;
            check_allowed_keys(*t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                                    "tmp_grad", "heat_flux_grad", "velocity"},
                            ctx, comm);

            if (t->contains("tmp_grad")) {
                d.tmp_grad = req_number(*t, "tmp_grad", ctx, comm);
            } else if (t->contains("heat_flux_grad")) {
                d.tmp_grad = req_number(*t, "heat_flux_grad", ctx, comm);
            } else {
                d.tmp_grad = 0.0; // default: adiabatic wall (dT/dn = 0)
            }

            if (t->contains("velocity")) {
                d.velocity = req_vec3(*t, "velocity", ctx, comm);
            } else {
                d.velocity = {0.0, 0.0, 0.0}; // default: stationary wall
            }
        } else if (type == "SUBSONIC_INLET") {
            d.type = bc::BCType::SubsonicInlet;
            if (!t->contains("p") && !t->contains("p_inf")) {
                // fictious pressure for parse_inflow_descriptor
                const_cast<toml::table*>(t)->insert_or_assign("p", constants::kIsaPressure);
            }
            parse_inflow_descriptor(*t, d, ctx, comm);
            
            if (d.inflow_mode == bc::InflowMode::MachAngles || d.inflow_mode == bc::InflowMode::MachDirection) {
                mpi::log_stat("WARNING: %s: Mach-based inflow mode selected. "
                                "Note: velocity magnitude is evaluated assuming Ideal Gas EOS kinematics.",
                                ctx.c_str());
            }
        } else if (type == "SUBSONIC_OUTLET") {
            d.type = bc::BCType::SubsonicOutlet;
            check_allowed_keys(*t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                                    "p", "p_outlet", "p_back"}, ctx, comm);

            if (t->contains("p_back")) {
                d.p = req_number(*t, "p_back", ctx, comm);
            } else if (t->contains("p_outlet")) {
                d.p = req_number(*t, "p_outlet", ctx, comm);
            } else if (t->contains("p")) {
                d.p = req_number(*t, "p", ctx, comm);
            } else {
                fail(comm, ctx + ": missing required backpressure ('p', 'p_back', or 'p_outlet')");
            }

            check_positive(d.p, "backpressure", ctx, comm);
        } else {
            fail(comm, ctx + ": unknown BC type '" + type + "'");
        } 

        ++idx;
    }

    for (std::size_t p = 0; p < n_patches; ++p) {
        if (seen[p] == 0) {
            fail(comm, "'" + path + "': no condition assigned to patch " +
                           std::to_string(p) + " ('" + mp.patches[p].name + "')");
        }
    }

    return out;
}

} // namespace cfd::solver::bc
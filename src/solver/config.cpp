#include "cfd/solver/config.hpp"

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <toml++/toml.hpp>

#include "cfd/mpi/log.hpp"

namespace cfd::solver {

namespace {

[[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

// Rank 0 reads the entire file into a std::string and broadcasts it to all ranks.
// Zero filesystem contention: only 1 rank accesses the disk.
[[nodiscard]] std::string broadcast_file_content(const std::string& path, const MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::string content;
    int content_size = 0;
    bool read_success = false;

    if (rank == 0) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            content = ss.str();
            content_size = static_cast<int>(content.size());
            read_success = true;
        }
    }

    // Broadcast file open status
    int status_int = read_success ? 1 : 0;
    MPI_Bcast(&status_int, 1, MPI_INT, 0, comm);

    if (status_int == 0) {
        fail(comm, "cannot open file '" + path + "' on Rank 0");
    }

    // Broadcast string length, allocate buffer on workers, broadcast string bytes
    MPI_Bcast(&content_size, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        content.resize(static_cast<std::size_t>(content_size));
    }
    MPI_Bcast(content.data(), content_size, MPI_CHAR, 0, comm);

    return content;
}

// Parses a TOML string in-memory or aborts with the parser's diagnostic.
[[nodiscard]] toml::table parse_in_memory_or_die(const std::string& content,
                                                 const std::string& source_path,
                                                 const MPI_Comm comm) {
    try {
        return toml::parse(content, source_path);
    } catch (const toml::parse_error& err) {
        fail(comm, "cannot parse '" + source_path + "': " + std::string(err.description()));
    }
}

[[nodiscard]] const toml::table* req_table(const toml::table& t, const char* key,
                                           const std::string& ctx, const MPI_Comm comm) {
    const toml::table* sub = t.get_as<toml::table>(key);
    if (sub == nullptr) {
        fail(comm, ctx + ": missing required section [" + key + "]");
    }
    return sub;
}

[[nodiscard]] double req_number(const toml::table& t, const char* key,
                                const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<double>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be a number");
    }
    return *val;
}

[[nodiscard]] double opt_number(const toml::table& t, const char* key, const double def) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        return def;
    }
    return node->value_or(def);
}

[[nodiscard]] std::int64_t req_integer(const toml::table& t, const char* key,
                                       const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<std::int64_t>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be an integer");
    }
    return *val;
}

[[nodiscard]] std::string req_string(const toml::table& t, const char* key,
                                     const std::string& ctx, const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const auto val = node->value<std::string>();
    if (!val.has_value()) {
        fail(comm, ctx + ": key '" + key + "' must be a string");
    }
    return *val;
}

[[nodiscard]] std::array<double, 3> req_vec3(const toml::table& t,
                                             const char* key,
                                             const std::string& ctx,
                                             const MPI_Comm comm) {
    const toml::node* node = t.get(key);
    if (node == nullptr) {
        fail(comm, ctx + ": missing required key '" + key + "'");
    }
    const toml::array* arr = node->as_array();
    if (arr == nullptr || arr->size() != 3) {
        fail(comm, ctx + ": key '" + key + "' must be an array of 3 numbers");
    }
    std::array<double, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto val = (*arr)[i].value<double>();
        if (!val.has_value()) {
            fail(comm, ctx + ": key '" + key + "' must be an array of 3 numbers");
        }
        out[i] = *val;
    }
    return out;
}

void check_allowed_keys(const toml::table& t,
                        const std::initializer_list<const char*> allowed,
                        const std::string& ctx, const MPI_Comm comm) {
    for (const auto& [k, v] : t) {
        (void)v;
        const std::string_view name = k.str();
        bool known = false;
        for (const char* a : allowed) {
            if (name == std::string_view(a)) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(comm, ctx + ": unknown key '" + std::string(name) + "'");
        }
    }
}

void check_positive(const double v, const char* what, const std::string& ctx,
                    const MPI_Comm comm) {
    if (!(v > 0.0)) {
        fail(comm, ctx + ": '" + what + "' must be positive (got " + std::to_string(v) + ")");
    }
}

} // anonymous namespace

SolverConfig parse_solver_config(const std::string& path, const MPI_Comm comm) {
    SolverConfig cfg;

    const std::string raw_content = broadcast_file_content(path, comm);
    const toml::table root = parse_in_memory_or_die(raw_content, path, comm);

    check_allowed_keys(root, {"flow", "initial", "numerics", "time", "output"},
                       "'" + path + "'", comm);

    { // [flow]
        const toml::table* t = req_table(root, "flow", path, comm);
        const std::string ctx = path + " [flow]";

        const std::string eos_name = req_string(*t, "eos", ctx, comm);
        if (eos_name == "IDEAL_GAS") {
            cfg.flow.type = EqOfStateType::IdealGas;
            check_allowed_keys(*t, {"eos", "gamma", "gas_constant"}, ctx, comm);
            cfg.flow.gamma = req_number(*t, "gamma", ctx, comm);
            cfg.flow.gas_constant = req_number(*t, "gas_constant", ctx, comm);
            check_positive(cfg.flow.gamma, "gamma", ctx, comm);
            check_positive(cfg.flow.gas_constant, "gas_constant", ctx, comm);
        } else {
            fail(comm, ctx + ": unsupported EOS model '" + eos_name + "' (available: IDEAL_GAS)");
        }
    }

    { // [initial]
        const toml::table* t = req_table(root, "initial", path, comm);
        const std::string ctx = path + " [initial]";
        check_allowed_keys(*t, {"rho", "pressure", "velocity"}, ctx, comm);
        cfg.init_rho = req_number(*t, "rho", ctx, comm);
        cfg.init_p = req_number(*t, "pressure", ctx, comm);
        cfg.init_velocity = req_vec3(*t, "velocity", ctx, comm);
        check_positive(cfg.init_rho, "rho", ctx, comm);
        check_positive(cfg.init_p, "pressure", ctx, comm);
    }

    { // [numerics]
        const toml::table* t = req_table(root, "numerics", path, comm);
        const std::string ctx = path + " [numerics]";
        check_allowed_keys(*t, {"flux", "reconstruction", "limiter", "venkat_k"}, ctx, comm);
        const std::string flux = req_string(*t, "flux", ctx, comm);
        if (flux == "HLLC") {
            cfg.flux = FluxType::HLLC;
        } else {
            fail(comm, ctx + ": unsupported flux '" + flux + "' (available: HLLC)");
        }
        const std::string reco = req_string(*t, "reconstruction", ctx, comm);
        if (reco == "FIRST_ORDER") {
            cfg.reconstruction = ReconType::FirstOrder;
        } else if (reco == "MUSCL") {
            cfg.reconstruction = ReconType::Muscl;
        } else {
            fail(comm, ctx + ": unsupported reconstruction '" + reco +
                           "' (available: FIRST_ORDER, MUSCL)");
        }
        // Limiter: required for MUSCL, optional otherwise (default VENKAT).
        std::string lim = "VENKAT";
        if (t->get("limiter") != nullptr) {
            lim = req_string(*t, "limiter", ctx, comm);
        } else if (cfg.reconstruction == ReconType::Muscl) {
            fail(comm, ctx + ": missing required key 'limiter' for MUSCL "
                           "(available: VENKAT, BARTH, VAN_ALBADA)");
        }
        
        if (lim == "VENKAT") {
            cfg.limiter = LimiterType::Venkatakrishnan;
        } else if (lim == "BARTH") {
            cfg.limiter = LimiterType::BarthJespersen;
        } else if (lim == "VAN_ALBADA") {
            cfg.limiter = LimiterType::VanAlbada;
        } else {
            fail(comm, ctx + ": unsupported limiter '" + lim +
                           "' (available: VENKAT, BARTH, VAN_ALBADA)");
        }
        cfg.limiter_venkat_k = opt_number(*t, "venkat_k", 0.5);
        check_positive(cfg.limiter_venkat_k, "venkat_k", ctx, comm);
    }

    { // [time]
        const toml::table* t = req_table(root, "time", path, comm);
        const std::string ctx = path + " [time]";
        check_allowed_keys(*t, {"scheme", "cfl", "max_iterations", "residual_tolerance"},
                           ctx, comm);
        const std::string scheme = req_string(*t, "scheme", ctx, comm);
        if (scheme == "FORWARD_EULER") {
            cfg.scheme = TimeScheme::ForwardEuler;
        } else if (scheme == "SSP_RK3") {
            cfg.scheme = TimeScheme::SspRk3;
        } else {
            fail(comm, ctx + ": unsupported scheme '" + scheme + "' (available: FORWARD_EULER, SSP_RK3)");
        }
        cfg.cfl = req_number(*t, "cfl", ctx, comm);
        check_positive(cfg.cfl, "cfl", ctx, comm);
        cfg.max_iterations = req_integer(*t, "max_iterations", ctx, comm);
        if (cfg.max_iterations < 1) {
            fail(comm, ctx + ": 'max_iterations' must be >= 1");
        }
        cfg.residual_tolerance = req_number(*t, "residual_tolerance", ctx, comm);
        check_positive(cfg.residual_tolerance, "residual_tolerance", ctx, comm);
    }

    { // [output]
        const toml::table* t = req_table(root, "output", path, comm);
        const std::string ctx = path + " [output]";
        check_allowed_keys(*t, {"directory", "field_interval", "residual_interval"},
                           ctx, comm);
        cfg.output_dir = req_string(*t, "directory", ctx, comm);
        cfg.field_interval = req_integer(*t, "field_interval", ctx, comm);
        cfg.residual_interval = req_integer(*t, "residual_interval", ctx, comm);
        if (cfg.field_interval < 0 || cfg.residual_interval < 1) {
            fail(comm, ctx + ": intervals must be >= 0 (field) / >= 1 (residual)");
        }
    }

    return cfg;
}

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

        const std::initializer_list<const char*> meta = {
            "patch_id", "type", "name", "cgns_type", "global_face_count"
        };

        if (type == "SUPERSONIC_INLET") {
            d.type = bc::BCType::SupersonicInlet;
            check_allowed_keys(*t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                                    "p", "t", "velocity"},
                               ctx, comm);
            d.p = req_number(*t, "p", ctx, comm);
            d.t = req_number(*t, "t", ctx, comm);
            d.velocity = req_vec3(*t, "velocity", ctx, comm);
            check_positive(d.p, "p", ctx, comm);
            check_positive(d.t, "t", ctx, comm);
        
        } else if (type == "SUPERSONIC_OUTLET") {
            d.type = bc::BCType::SupersonicOutlet;
            check_allowed_keys(*t, meta, ctx, comm);
        
        } else if (type == "SLIP_WALL") {
            d.type = bc::BCType::SlipWall;
            check_allowed_keys(*t, meta, ctx, comm);
        
        } else if (type == "SYMMETRY") {
            d.type = bc::BCType::Symmetry;
            check_allowed_keys(*t, meta, ctx, comm);
        
        } else if (type == "FARFIELD") {
            d.type = bc::BCType::Farfield;
            check_allowed_keys(*t, {"patch_id", "type", "name", "cgns_type", "global_face_count",
                                    "velocity_inf", "p_inf", "t_inf", "alpha", "beta"},
                               ctx, comm);
            d.p_inf = req_number(*t, "p_inf", ctx, comm);
            d.t_inf = req_number(*t, "t_inf", ctx, comm);
            d.velocity_inf = req_vec3(*t, "velocity_inf", ctx, comm);
            check_positive(d.p_inf, "p_inf", ctx, comm);
            check_positive(d.t_inf, "t_inf", ctx, comm);
        
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

} // namespace cfd::solver
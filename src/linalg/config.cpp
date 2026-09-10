#include "cfd/linalg/config.hpp"


#include "cfd/io/toml/toml_utilities.hpp"
#include "cfd/mpi/log.hpp"
#include "toml++/toml.hpp"

using namespace cfd::io::toml_utils;

namespace cfd::linalg {

namespace {
    
[[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

}

SolverParams parse_solver_config(const std::string& path, const MPI_Comm comm) {
    SolverParams result;

    const std::string raw_content = broadcast_file_content(path, comm);
    const toml::table root = parse_in_memory_or_die(raw_content, path, comm);
    const std::string ctx = path;

    check_allowed_keys(root, {"solver", "preconditioner", "rel_tol", "abs_tol",
                                         "max_iter", "verbosity", "res_verify"}, "'" + path + "'", comm);
        
    std::string type = opt_string(root, "solver", "BICGSTAB");
    if (type == "BICGSTAB") {
        result.type = SolverType::BICGSTAB;
    } else {
        fail(comm, ctx + ": unsupported solver type '" + type + "' (available: BICGSTAB)");
    }

    std::string precond_type = opt_string(root, "preconditioner", "None");
    if (precond_type == "None") {
        result.precond_type = PreconditionerType::None;
    } else if (precond_type == "SGS") {
        result.precond_type = PreconditionerType::SGS;
    } else {
        fail(comm, ctx + ": unsupported solver type '" + type + "' (available: BICGSTAB)");
    }

    result.relative_tolerance = opt_number(root, "rel_tol", 1e-1);
    result.absolute_tolerance = opt_number(root, "abs_tol", 1e-30);
    result.max_iterations = static_cast<int>(opt_integer(root, "max_iter", 100));

    std::string verbosity = opt_string(root, "verbosity", "Silent");
    if (verbosity == "Silent") {
        result.verbosity = Verbosity::Silent;
    } else if (verbosity == "Summary") {
        result.verbosity = Verbosity::Summary;
    } else if (verbosity == "Verbose") {
        result.verbosity = Verbosity::Verbose;
    } else {
        fail(comm, ctx + ": unsupported solver type '" + type + "' (available: BICGSTAB)");
    }

    return result;
}

} // namespace cfd::linalg
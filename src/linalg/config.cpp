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

    const toml::node* node = root.get("linalg");
    if (node != nullptr) {
        const auto* t = node->as_table();
        if (t == nullptr) {
            fail(comm, path + ": [linalg] must be a table");
        }
        const std::string ctx = path + " [linalg]";
        check_allowed_keys(*t, {"solver", "preconditioner", "rel_tol", "abs_tol",
                                        "max_iter", "verbosity", "res_verify"}, ctx, comm);
        
        std::string type = opt_string(*t, "solver", "BICGSTAB");
        if (type == "BICGSTAB") {
            result.type = SolverType::BICGSTAB;
        } else {
            fail(comm, ctx + ": unsupported solver type '" + type + "' (available: BICGSTAB)");
        }

        std::string precond_type = opt_string(*t, "preconditioner", "None");
        if (precond_type == "None") {
            result.precond_type = PreconditionerType::None;
        } else if (precond_type == "SGS") {
            result.precond_type = PreconditionerType::SGS;
        } else {
            fail(comm, ctx + ": unsupported preconditioner type '" + precond_type + "' (available: None, SGS)");
        }

        result.relative_tolerance = opt_number(*t, "rel_tol", 1e-1);
        result.absolute_tolerance = opt_number(*t, "abs_tol", 1e-30);
        result.max_iterations = static_cast<int>(opt_integer(*t, "max_iter", 100));

        check_positive(result.relative_tolerance, "rel_tol", ctx, comm);
        check_positive(result.absolute_tolerance, "abs_tol", ctx, comm);
        check_positive(result.max_iterations, "max_iter", ctx, comm);

        std::string verbosity = opt_string(*t, "verbosity", "Silent");
        if (verbosity == "Silent") {
            result.verbosity = Verbosity::Silent;
        } else if (verbosity == "Summary") {
            result.verbosity = Verbosity::Summary;
        } else if (verbosity == "Verbose") {
            result.verbosity = Verbosity::Verbose;
        } else {
            fail(comm, ctx + ": unsupported verbosity type '" + verbosity + "' (available: Silent, Summary, Verbose)");
        }
    
    } else {
        fail(comm, path + ": file must define [linalg] module");
    }
        

    return result;
}

} // namespace cfd::linalg
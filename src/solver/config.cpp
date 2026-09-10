#include "cfd/solver/config.hpp"

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "cfd/io/toml/toml_utilities.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/solver/gradient/gradient_manager.hpp"

using namespace cfd::io::toml_utils;

namespace cfd::solver {

namespace {
[[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

} // anonymous namespace

SolverConfig parse_solver_config(const std::string& path, const MPI_Comm comm) {
    SolverConfig cfg;

    const std::string raw_content = broadcast_file_content(path, comm);
    const toml::table root = parse_in_memory_or_die(raw_content, path, comm);

    check_allowed_keys(root, {"flow", "initial", "numerics", "time", "output", "turbulence", "linalg"},
                    "'" + path + "'", comm);

    { // [flow]
        const toml::table* t = req_table(root, "flow", path, comm);
        const std::string ctx = path + " [flow]";
        
        const std::string eos_name = req_string(*t, "eos", ctx, comm);
        const std::string flow_model_name = req_string(*t, "flow_model", ctx, comm);
        
        std::vector<std::string> allowed_keys = {"eos", "flow_model"};

        // eos parsing
        if (eos_name == "IDEAL_GAS") {
            cfg.flow.type = EqOfStateType::IdealGas;
            allowed_keys.push_back("gamma");
            allowed_keys.push_back("gas_constant");

            cfg.flow.gamma = req_number(*t, "gamma", ctx, comm);
            cfg.flow.gas_constant = req_number(*t, "gas_constant", ctx, comm);
            check_positive(cfg.flow.gamma, "gamma", ctx, comm);
            check_positive(cfg.flow.gas_constant, "gas_constant", ctx, comm);
        } else {
            fail(comm, ctx + ": unsupported EOS model '" + eos_name + "' (available: IDEAL_GAS)");
        }

        // flow model parsing
        if (flow_model_name == "INVISCID_FLOW") {
            cfg.flow_model = FlowModel::InviscidFlow;

        } else if (flow_model_name == "VISCOUS_FLOW") {
            cfg.flow_model = FlowModel::ViscousFlow;
            allowed_keys.push_back("prandtl");

            cfg.prandtl = req_number(*t, "prandtl", ctx, comm);
            check_positive(cfg.prandtl, "prandtl", ctx, comm);
        } else {
            fail(comm, ctx + ": unsupported Flow Model '" + flow_model_name + "' (available: INVISCID_FLOW, VISCOUS_FLOW)");
        }

        check_allowed_keys(*t, allowed_keys, ctx, comm);
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
        check_allowed_keys(*t, {"flux", "reconstruction", "limiter", "venkat_k", "gradient"}, ctx, comm);

        const std::string flux = req_string(*t, "flux", ctx, comm);
        if (flux == "HLLC") {
            cfg.flux = FluxType::HLLC;
        } else {
            fail(comm, ctx + ": unsupported flux '" + flux + "' (available: HLLC)");
        }

        const std::string reco = req_string(*t, "reconstruction", ctx, comm);
        if (reco == "FIRST_ORDER") {
            cfg.reconstruction = ReconType::FirstOrder;
            cfg.limiter = LimiterType::None;

        } else if (reco == "MUSCL") {
            cfg.reconstruction = ReconType::Muscl;
            const std::string lim = req_string(*t, "limiter", ctx, comm);
            if (lim == "VENKAT") {
                cfg.limiter = LimiterType::Venkatakrishnan;
            } else if (lim == "BARTH") {
                cfg.limiter = LimiterType::BarthJespersen;
            } else if (lim == "VAN_ALBADA") {
                cfg.limiter = LimiterType::VanAlbada;
            } else {
                fail(comm, ctx + ": invalid limiter '" + lim + 
                            "' for MUSCL (available: VENKAT, BARTH, VAN_ALBADA)");
            }

        } else if (reco == "MUSCL_DIRECTIONAL") {
            cfg.reconstruction = ReconType::MusclDirectional;
            const std::string lim = req_string(*t, "limiter", ctx, comm);
            if (lim == "MINMOD_1D") {
                cfg.limiter = LimiterType::Minmod1D;
            } else if (lim == "VAN_ALBADA_1D") {
                cfg.limiter = LimiterType::VanAlbada1D;
            } else {
                fail(comm, ctx + ": invalid limiter '" + lim + 
                            "' for MUSCL_DIRECTIONAL (available: MINMOD_1D, VAN_ALBADA_1D)");
            }

        } else {
            fail(comm, ctx + ": unsupported reconstruction '" + reco +
                           "' (available: FIRST_ORDER, MUSCL, MUSCL_DIRECTIONAL)");
        }

        cfg.limiter_venkat_k = opt_number(*t, "venkat_k", 0.5);
        check_positive(cfg.limiter_venkat_k, "venkat_k", ctx, comm);

        const std::string gradient_type = opt_string(*t, "gradient", "GREEN_GAUSS_FACE");
        if (gradient_type == "GREEN_GAUSS_FACE") {
            cfg.gradient = gradient::GradientType::GreenGaussFace;
        } else if (gradient_type == "GREEN_GAUSS_CELL") {
            cfg.gradient = gradient::GradientType::GreenGaussCell;
        } else if (gradient_type == "LEAST_SQUARES_FACE") {
            cfg.gradient = gradient::GradientType::LeastSquaresCellNode;
        } else if (gradient_type == "LEAST_SQUARES_NODE") {
            cfg.gradient = gradient::GradientType::LeastSquaresCellNode;
        } else {
            fail(comm, ctx + ": unsupported gradient '" + gradient_type +
                           "' (available: GREEN_GAUSS_FACE, GREEN_GAUSS_CELL, LEAST_SQUARES_FACE, LEAST_SQUARES_NODE)");
        }
        
    }

    { // [turbulence] — optional physics module selection
        const toml::node* node = root.get("turbulence");
        if (node != nullptr) {
            const auto* t = node->as_table();
            if (t == nullptr) {
                fail(comm, path + ": [turbulence] must be a table");
            }
            const std::string ctx = path + " [turbulence]";
            check_allowed_keys(*t, {"model", "nu_inf_ratio", "max_distance_sweeps",
                                    "distance_tolerance"}, ctx, comm);

            const std::string model = req_string(*t, "model", ctx, comm);
            if (model == "SA") {
                cfg.turbulence_model = TurbulenceModel::SA;
                cfg.turbulence.enabled = true;
            } else {
                fail(comm, ctx + ": unsupported turbulence model '" + model +
                            "' (available: SA)");
            }

            cfg.turbulence.nu_inf_ratio = opt_number(*t, "nu_inf_ratio", 3.0);
            check_positive(cfg.turbulence.nu_inf_ratio, "nu_inf_ratio", ctx, comm);

            cfg.turbulence.max_distance_sweeps =
                static_cast<int>(opt_integer(*t, "max_distance_sweeps", 500));

            cfg.turbulence.distance_tolerance =
                opt_number(*t, "distance_tolerance", 1.0e-8);
            check_positive(cfg.turbulence.distance_tolerance, "distance_tolerance", ctx, comm);

            if (cfg.turbulence.enabled && !(cfg.flow_model == FlowModel::ViscousFlow)) {
                fail(comm, ctx + ": turbulence requires [numerics] viscous = true");
            }
        }
    }

    { // [time]
        const toml::table* t = req_table(root, "time", path, comm);
        const std::string ctx = path + " [time]";
        check_allowed_keys(
            *t, {"scheme", "cfl", "max_iterations", "residual_tolerance"},
            ctx, comm);
        const std::string scheme = req_string(*t, "scheme", ctx, comm);
        if (scheme == "FORWARD_EULER") {
            cfg.scheme = TimeScheme::ForwardEuler;
        } else if (scheme == "SSP_RK3") {
            cfg.scheme = TimeScheme::SspRk3;
        } else if (scheme == "BACKWARD_EULER") {
            cfg.scheme = TimeScheme::BackwardEuler;
        } else {
            fail(comm, ctx + ": unsupported scheme '" + scheme +
                           "' (available: FORWARD_EULER, SSP_RK3, BACKWARD_EULER)");
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

    { // [linalg] - optional linear algebra module section
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
                cfg.linear_solver_params.type = linalg::SolverType::BICGSTAB;
            } else {
                fail(comm, ctx + ": unsupported solver type '" + type + "' (available: BICGSTAB)");
            }

            std::string precond_type = opt_string(*t, "preconditioner", "None");
            if (precond_type == "None") {
                cfg.linear_solver_params.precond_type = linalg::PreconditionerType::None;
            } else if (precond_type == "SGS") {
                cfg.linear_solver_params.precond_type = linalg::PreconditionerType::SGS;
            } else {
                fail(comm, ctx + ": unsupported preconditioner type '" + precond_type + "' (available: None, SGS)");
            }

            cfg.linear_solver_params.relative_tolerance = opt_number(*t, "rel_tol", 1e-1);
            cfg.linear_solver_params.absolute_tolerance = opt_number(*t, "abs_tol", 1e-30);
            cfg.linear_solver_params.max_iterations = static_cast<int>(opt_integer(*t, "max_iter", 100));

            std::string verbosity = opt_string(*t, "verbosity", "Silent");
            if (verbosity == "Silent") {
                cfg.linear_solver_params.verbosity = linalg::Verbosity::Silent;
            } else if (verbosity == "Summary") {
                cfg.linear_solver_params.verbosity = linalg::Verbosity::Summary;
            } else if (verbosity == "Verbose") {
                cfg.linear_solver_params.verbosity = linalg::Verbosity::Verbose;
            } else {
                fail(comm, ctx + ": unsupported verbosity type '" + verbosity + "' (available: Silent, Summary, Verbose)");
            }

            check_positive(cfg.linear_solver_params.relative_tolerance, "rel_tol", ctx, comm);
            check_positive(cfg.linear_solver_params.absolute_tolerance, "abs_tol", ctx, comm);
            check_positive(cfg.linear_solver_params.max_iterations, "max_iter", ctx, comm);
        }
    }

    return cfg;
}

} // namespace cfd::solver
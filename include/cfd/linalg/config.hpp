#pragma once

#include <mpi.h>

namespace cfd::linalg {
  
enum class SolverType { BICGSTAB };

enum class PreconditionerType { None, SGS};

enum class Verbosity { Silent, Summary, Verbose };

struct SolverParams {
    SolverType type = SolverType::BICGSTAB;
    PreconditionerType precond_type = PreconditionerType::None;
    
    // Convergence test: ||r||_2 <= max(relative_tolerance * ||b||_2, absolute_tolerance).
    double relative_tolerance = 1.0e-1;
    double absolute_tolerance = 1.0e-15;
    int max_iterations = 500;
    Verbosity verbosity = Verbosity::Silent;
    // Recompute the true residual ||b - A x||_2 once after the loop and report
    // it in IterationResult (costs one extra SpMV per solve).
    bool verify_final_residual = false;
};

SolverParams parse_solver_config(const std::string& path, const MPI_Comm comm);

}  // namespace cfd::linalg

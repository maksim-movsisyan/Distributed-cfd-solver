// Distributed linear algebra module: shared types, solver configuration and
// error handling. The module is self-contained: it depends only on MPI and the
// index vocabulary from cfd/core/types.hpp (no CFD-level concepts here).
#pragma once

#include <mpi.h>

#include <string>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"

namespace cfd::linalg {

// The module reuses the project-wide index vocabulary:
//   GlobalIndex — unique row / column id across all ranks,
//   LocalIndex  — slot in rank-local arrays (owned block first, ghosts after).

// --- Error handling ----------------------------------------------------------

// Reports `what` from the calling rank and aborts the whole job (MPI_Abort), so
// no rank is left waiting for communication with the failed one.
[[noreturn]] void fatal(MPI_Comm comm, const std::string& what);
[[noreturn]] void fatal(MPI_Comm comm, const char* what);

// Precondition-style guard: aborts with `message` when `condition` is false.
// Prefer const char* literals so the happy path stays allocation-free.
template <typename Message>
inline void check(bool condition, MPI_Comm comm, Message&& message) {
    if (!condition) fatal(comm, std::forward<Message>(message));
}

// --- Iterative solvers -------------------------------------------------------

enum class SolverStatus {
    Converged,     // stopping criterion reached
    Diverged,      // breakdown (rho / alpha / omega) or non-finite residual
    MaxIterations, // iteration budget exhausted
};

enum class Verbosity { Silent, Summary, Verbose };

struct SolverParams {
    // Convergence test: ||r||_2 <= max(relative_tolerance * ||b||_2, absolute_tolerance).
    double relative_tolerance = 1.0e-8;
    double absolute_tolerance = 1.0e-30;
    int max_iterations = 500;
    Verbosity verbosity = Verbosity::Summary;
    // Recompute the true residual ||b - A x||_2 once after the loop and report
    // it in IterationResult (costs one extra SpMV per solve).
    bool verify_final_residual = true;
};

struct IterationResult {
    SolverStatus status = SolverStatus::MaxIterations;
    int iterations = 0;
    double initial_residual = 0.0;         // ||b - A x0||_2
    double final_residual = 0.0;           // recursive residual norm at exit
    double true_final_residual = -1.0;     // ||b - A x||_2, filled when verify is on
    std::vector<double> residual_history;  // ||r_k||_2 / ||b||_2 per iteration
};

namespace detail {

// MPI datatype matching the (switchable) GlobalIndex width.
inline MPI_Datatype mpi_index_type() {
    return sizeof(GlobalIndex) == 8 ? MPI_INT64_T : MPI_INT32_T;
}

}  // namespace detail

}  // namespace cfd::linalg

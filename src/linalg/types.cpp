#include "cfd/linalg/types.hpp"

#include <cstdio>
#include <cstdlib>

namespace cfd::linalg {

void fatal(MPI_Comm comm, const std::string& what) { fatal(comm, what.c_str()); }

void fatal(MPI_Comm comm, const char* what) {
    int rank = 0;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &rank);
    }
    std::fprintf(stderr, "[Rank %d] cfd_linalg fatal: %s\n", rank, what);
    std::fflush(stderr);
    MPI_Abort(comm != MPI_COMM_NULL ? comm : MPI_COMM_WORLD, 1);
    std::abort();  // MPI_Abort is not declared noreturn in the MPI bindings
}

}  // namespace cfd::linalg

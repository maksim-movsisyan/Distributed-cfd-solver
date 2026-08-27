#include "cfd/mpi/log.hpp"

#include <mpi.h>
#include <cstdarg>
#include <cstdio>

int g_verbose = 0;

namespace cfd::mpi {

static int mpi_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

void log_init(int verbose) { g_verbose = verbose; }
bool log_verbose() { return g_verbose >= 1; }

static void vpr(int rank, const char* prefix, const char* fmt, va_list ap) {
    std::fprintf(stderr, "%s", prefix);
    if (rank >= 0) std::fprintf(stderr, "[r%d] ", rank);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
}

void log_info(const char* fmt, ...) {
    if (mpi_rank() != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vpr(-1, "", fmt, ap);
    va_end(ap);
}

void log_stat(const char* fmt, ...) {
    if (mpi_rank() != 0 || g_verbose < 1) return;
    va_list ap;
    va_start(ap, fmt);
    vpr(-1, "", fmt, ap);
    va_end(ap);
}

void log_rank(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vpr(mpi_rank(), "", fmt, ap);
    va_end(ap);
}

void log_warn_rank(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vpr(mpi_rank(), "WARNING: ", fmt, ap);
    va_end(ap);
}

void fatal(MPI_Comm comm, const std::string& what) {
    int r = 0;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &r);
    }
    std::fprintf(stderr, "[Rank %d] CGNS Reader Fatal Error: %s\n", r, what.c_str());
    std::fflush(stderr);
    MPI_Abort(comm != MPI_COMM_NULL ? comm : MPI_COMM_WORLD, 1);
}
} //namespace cfd::mpi
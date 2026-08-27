// Simple logger: rank-0 printing plus a controllable verbose mode.
#pragma once

#include <string>
#include <mpi.h>

extern int g_verbose;  // 0 = quiet, 1 = statistics, 2 = detailed

namespace cfd::mpi {

void log_init(int verbose);
bool log_verbose();

// always printed (rank 0 only)
void log_info(const char* fmt, ...);
// printed when verbose >= 1 (rank 0 only)
void log_stat(const char* fmt, ...);
// printed from any rank with a [rank] prefix
void log_rank(const char* fmt, ...);
// warning (any rank)
void log_warn_rank(const char* fmt, ...);
// fatal error
void fatal(MPI_Comm comm, const std::string& what);
} //namespace cfd::mpi
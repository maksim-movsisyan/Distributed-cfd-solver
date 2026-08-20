#pragma once
// Simple logger: rank-0 printing plus a controllable verbose mode.
#include <cstdio>
#include <string>

extern int g_verbose;  // 0 = quiet, 1 = statistics, 2 = detailed

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

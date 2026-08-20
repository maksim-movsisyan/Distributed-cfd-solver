#pragma once
// Small MPI helpers: block distribution and all-to-all for vectors of PODs.
#include <mpi.h>

#include <vector>

// Balanced block distribution of n elements over p parts:
// displs[i]..displs[i+1] — rank i owns displs[i]..displs[i+1] (displs has size p+1).
std::vector<long long> block_displ(long long n, int p);

// Owner of element g under the block distribution.
int block_owner(long long g, const std::vector<long long>& displ);

// All-to-all of POD structs (T must be trivially copyable).
// in[i] is the buffer for rank i; returns out[i], the buffer from rank i.
template <class T>
std::vector<std::vector<T>> alltoall_vec(const std::vector<std::vector<T>>& in);

// Global sum/min/max (short wrappers).
long long ll_sum(long long local);
long long ll_min(long long local);
long long ll_max(long long local);
double d_sum(double local);
double d_min(double local);
double d_max(double local);

#include "mpi_util_impl.hpp"

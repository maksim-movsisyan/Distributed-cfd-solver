#include "cfd/mpi/mpi_util.hpp"

std::vector<long long> block_displ(long long n, int p) {
    std::vector<long long> d(p + 1, 0);
    for (int i = 0; i < p; ++i) d[i + 1] = d[i] + (n / p) + ((n % p) > i ? 1 : 0);
    return d;
}

int block_owner(long long g, const std::vector<long long>& displ) {
    // displ is strictly increasing, g in [displ[0], displ.back())
    int lo = 0, hi = static_cast<int>(displ.size()) - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (displ[mid] <= g)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

long long ll_sum(long long local) {
    long long s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return s;
}
long long ll_min(long long local) {
    long long s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
    return s;
}
long long ll_max(long long local) {
    long long s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
    return s;
}
double d_sum(double local) {
    double s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return s;
}
double d_min(double local) {
    double s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return s;
}
double d_max(double local) {
    double s = local;
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return s;
}

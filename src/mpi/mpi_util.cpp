#include "cfd/mpi/mpi_util.hpp"

namespace cfd::mpi {
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
}


// solver_mesh_check: read the custom HDF5 exactly as the solver will, with
// full verification and metadata printing.
//
// Usage:
//   mpirun -np N loader <part.h5> [-v|--verbose] [--dump-vtu <dir>]
#include <mpi.h>

#include <cstdio>
#include <string>

#include "cfd/io/solver_mesh/h5util.hpp"
#include "cfd/io/solver_mesh/loader.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mpi/log.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string in, vtudir;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
            verbose = 1;
        else if (a == "-vv")
            verbose = 2;
        else if (a == "--dump-vtu" && i + 1 < argc)
            vtudir = argv[++i];
        else if (in.empty())
            in = a;
    }
    if (in.empty()) {
        if (rank == 0)
            std::fprintf(stderr,
                         "usage: mpirun -np N solver_mesh_check <part.h5> "
                         "[-v] [--dump-vtu <dir>]\n");
        MPI_Finalize();
        return 1;
    }
    log_init(verbose);

    const double t0 = MPI_Wtime();
    H5Obj file = h5_open_file_rdonly(in);

    GlobalMeta gm = load_global_meta(file);
    MeshPart mp;
    load_partition(file, rank, gm, mp);
    const double t1 = MPI_Wtime();

    print_mesh_stats(mp, gm);

    const bool ok = verify_partition(mp, gm);
    if (rank == 0) {
        log_info("Verification: %s", ok ? "OK — the file is consistent" : "FAILED (see above)");
        log_info("load time: %.3f s", t1 - t0);
    }

    if (!vtudir.empty()) write_vtu(mp, vtudir, "loaded");

    MPI_Finalize();
    return ok ? 0 : 2;
}

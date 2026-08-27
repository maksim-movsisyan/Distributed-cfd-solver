// solver_mesh_check: read the custom HDF5 exactly as the solver will, with
// full verification and metadata printing.
//
// Usage:
//   mpirun -np N loader <part.h5> [-v|--verbose] [--dump-vtu <dir>]
#include <mpi.h>

#include <cstdio>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "cfd/io/solver_mesh/hdf5_reader.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/validate.hpp"
#include "cfd/mpi/log.hpp"

int main(int argc, char** argv) {
    // MPI initialization 
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_status = 
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    if (init_status != MPI_SUCCESS) { 
        std::cerr << "MPI_Init_thread failed\n"; return EXIT_FAILURE;
    }


    // get local rank index and total process number
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    // check if target thread level is avaliable
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI did not provide the requested thread level\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    // parsing arguments
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
    cfd::mpi::log_init(verbose);
    cfd::mpi::log_info("solver_mesh_check: input=%s, ranks=%d", in.c_str(), nprocs);

    double t0 = MPI_Wtime();
    cfd::mesh::MeshPart mp;
    cfd::io::solver_mesh::import_mesh_hdf5(mp, in, MPI_COMM_WORLD);

    // Exhaustive Sanity Check & Global Diagnostics  
    cfd::mesh::validate_and_log_meshpart(mp);

    if (!vtudir.empty()) cfd::io::vtk::write_vtu(mp, vtudir, "loaded", MPI_COMM_WORLD);

    double t1 = MPI_Wtime() - t0;
    if (rank == 0) std::fprintf(stderr, "Total executional time = %.5f sec\n", t1);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

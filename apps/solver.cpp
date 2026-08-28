// solver: compressible Euler solver on the partitioned HDF5 mesh container.
//
// Usage:
//   mpirun -np N solver <mesh.h5> <solver.toml> <bc.toml> [-v|-vv]
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "cfd/io/solver_mesh/hdf5_reader.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/validate.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/solver/config.hpp"
#include "cfd/solver/solver.hpp"

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
    std::string mesh_file, solver_cfg_file, bc_cfg_file;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
            verbose = 1;
        else if (a == "-vv")
            verbose = 2;
        else if (mesh_file.empty())
            mesh_file = a;
        else if (solver_cfg_file.empty())
            solver_cfg_file = a;
        else if (bc_cfg_file.empty())
            bc_cfg_file = a;
        else {
            if (rank == 0)
                std::fprintf(stderr,
                             "usage: mpirun -np N solver <mesh.h5> <solver.toml> "
                             "<bc.toml> [-v|-vv]\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (mesh_file.empty() || solver_cfg_file.empty() || bc_cfg_file.empty()) {
        if (rank == 0)
            std::fprintf(stderr,
                         "usage: mpirun -np N solver <mesh.h5> <solver.toml> "
                         "<bc.toml> [-v|-vv]\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cfd::mpi::log_init(verbose);
    cfd::mpi::log_info("solver: mesh=%s config=%s bc=%s ranks=%d",
                       mesh_file.c_str(), solver_cfg_file.c_str(),
                       bc_cfg_file.c_str(), nprocs);

    const double t0 = MPI_Wtime();
    
    // Import partitioned mesh + mesh validation
    cfd::mesh::MeshPart mp;
    cfd::io::solver_mesh::import_mesh_hdf5(mp, mesh_file, MPI_COMM_WORLD);
    cfd::mesh::validate_and_log_meshpart(mp);

    // Read solver config and boundary conditions file
    const cfd::solver::SolverConfig cfg =
        cfd::solver::parse_solver_config(solver_cfg_file, MPI_COMM_WORLD);
    const cfd::solver::BoundaryConfig bcfg =
        cfd::solver::parse_boundary_config(bc_cfg_file, mp, MPI_COMM_WORLD);

    const int status = cfd::solver::run_solver(cfg, bcfg, mp, MPI_COMM_WORLD);
        
    if (rank == 0)
        std::fprintf(stderr, "Total executional time = %.5f sec\n",
                     MPI_Wtime() - t0);
    MPI_Finalize();
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

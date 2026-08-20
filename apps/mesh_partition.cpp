// mesh_partition: parallel CGNS preprocessing -> custom HDF5 + BC file + VTU.
//
// Usage:
//   mpirun -np N preproc <in.cgns> <out.h5> [--bc <bc.txt>] [--vtu <dir>]
//                        [--verbose|-v]
#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "cfd/io/solver_mesh/loader.hpp"
#include "cfd/io/solver_mesh/writer.hpp"
#include "cfd/io/vtk/vtu.hpp"
#include "cfd/mesh/bc.hpp"
#include "cfd/mesh/cgns_reader.hpp"
#include "cfd/mesh/faces.hpp"
#include "cfd/mesh/geometry.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"
#include "cfd/partition/partition.hpp"

static void usage() {
    std::fprintf(stderr,
                 "usage: mpirun -np N mesh_partition <in.cgns> <out.h5> "
                 "[--bc <bc.txt>] [--vtu <dir>] [-v|--verbose]\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string in, out, bcfile, vtudir;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
            verbose = 1;
        else if (a == "-vv")
            verbose = 2;
        else if (a == "--bc" && i + 1 < argc)
            bcfile = argv[++i];
        else if (a == "--vtu" && i + 1 < argc)
            vtudir = argv[++i];
        else if (in.empty())
            in = a;
        else if (out.empty())
            out = a;
        else {
            if (rank == 0) usage();
            MPI_Finalize();
            return 1;
        }
    }
    if (in.empty() || out.empty()) {
        if (rank == 0) usage();
        MPI_Finalize();
        return 1;
    }
    if (bcfile.empty()) bcfile = out + ".bc";
    if (vtudir.empty()) vtudir = ".";
    log_init(verbose);

    double t0 = MPI_Wtime();
    log_info("mesh_partition: input=%s output=%s ranks=%d", in.c_str(), out.c_str(), nprocs);

    // --- 0. Self-check of the canonical face tables ---
    if (!validate_face_tables()) {
        if (rank == 0)
            std::fprintf(stderr,
                         "ERROR: canonical face table self-check "
                         "FAILED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    log_stat("Canonical face table self-check: OK");

    // --- 1. Parallel CGNS reading ---
    RawMesh* m = read_cgns_parallel(in);

    // --- 2. Distributed face construction ---
    std::vector<FaceRec> faces;
    DualGraph graph;
    FaceStats fst;
    build_faces(*m, faces, graph, fst);

    // --- 3. Partitioning + part-to-rank mapping ---
    PartitionResult pr;
    partition_cells(*m, graph, pr, /*num_threads=*/2);

    // --- 4. Migration, ghost layer, renumbering ---
    MeshPart mp;
    build_local_mesh(*m, faces, pr, mp);

    // --- 5. Geometry ---
    const bool geom_ok = compute_geometry(mp);

    // --- 6. Boundary conditions ---
    const bool bc_ok = match_boundaries(*m, mp, faces);

    // --- 7. Sanity check + writing ---
    std::string err;
    if (!meshpart_sane(mp, err)) {
        log_rank("MeshPart invariants violated: %s", err.c_str());
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    GlobalMeta gm;
    gm.nprocs = nprocs;
    gm.n_cells_g = mp.n_cells_g;
    gm.n_faces_g = mp.n_faces_g;
    gm.n_nodes_g = mp.n_nodes_g;
    gm.n_bfaces_g = mp.n_bfaces_g;
    gm.n_flipped = mp.n_flipped;
    gm.n_ghost_total = ll_sum(mp.n_cells - mp.n_own);
    for (int d = 0; d < 3; ++d) {
        gm.bbox_lo[d] = mp.bbox_lo[d];
        gm.bbox_hi[d] = mp.bbox_hi[d];
    }
    {
        double vs = 0.0;
        for (int i = 0; i < mp.n_own; ++i) vs += mp.cell_volume[i];
        gm.total_volume = d_sum(vs);
    }
    for (auto& p : mp.patches) {
        gm.patch_names.push_back(p.name);
        gm.patch_types.push_back(p.cgns_type);
    }

    write_mesh_file(out, mp, gm);
    if (rank == 0) write_bc_config(mp, bcfile);
    write_vtu(mp, vtudir, "part");

    // --- 8. Final statistics ---
    const double t1 = MPI_Wtime();
    log_info("=== Preprocessing summary ===");
    log_info("cells %lld, faces %lld (boundary %lld), nodes %lld", gm.n_cells_g, gm.n_faces_g,
             gm.n_bfaces_g, gm.n_nodes_g);
    log_info("ghost cells total %lld (%.2f%% of owned)", gm.n_ghost_total,
             gm.n_cells_g ? 100.0 * gm.n_ghost_total / gm.n_cells_g : 0.0);
    log_info("volume sum %.12g; fixed orientations %lld", gm.total_volume, gm.n_flipped);
    log_stat("edge_cut=%lld, imbalance=%.3f", pr.edge_cut, pr.imbalance);
    log_info("geometry: %s", geom_ok ? "OK (all normals outward)" : "PROBLEMS (see warnings)");
    log_info("BCs: %s",
             bc_ok ? "all boundary faces covered" : "uncovered faces / BC elements (see warnings)");
    {
        // Patches: global face counts.
        std::vector<long long> per(mp.patches.size(), 0);
        for (int i = 0; i < mp.n_faces; ++i)
            if (mp.face_patch[i] >= 0) ++per[mp.face_patch[i]];
        std::vector<long long> gper(per.size());
        MPI_Reduce(per.data(), gper.data(), (int)per.size(), MPI_LONG_LONG_INT, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (rank == 0)
            for (size_t p = 0; p < gper.size(); ++p)
                log_info("  patch %zu '%s': %lld faces", p, mp.patches[p].name.c_str(), gper[p]);
    }
    log_info("preprocessing time: %.2f s", t1 - t0);
    log_info("BC file: %s", bcfile.c_str());

    delete m;
    MPI_Finalize();
    return 0;
}

#include "cfd/io/vtk/vtu.hpp"

#include <mpi.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "cfd/mpi/log.hpp"

namespace {

// One output cell: either an owned volume cell (data = global cell id) or a
// boundary face (data = patch id).
struct OutCell {
    uint8_t type;  // CellType
    int32_t nodes[8];
    int nn;
    int32_t data;
};

// Write one VTU piece (all ranks write their own file) + the .pvtu
// collection from rank 0. `data_name` is the CellData field carried by the
// piece: "global_id" for the volume piece, "patch" for the boundary piece.
void write_piece(const MeshPart& mp, const std::string& outdir, const std::string& stem,
                 const std::vector<OutCell>& cells, const char* data_name) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int nc = static_cast<int>(cells.size());
    const int npts = mp.n_nodes;  // the point array is simply the local nodes

    char path[512];
    std::snprintf(path, sizeof path, "%s/%s_%05d.vtu", outdir.c_str(), stem.c_str(), rank);
    FILE* f = std::fopen(path, "w");
    if (!f) {
        log_warn_rank("could not open %s", path);
        return;
    }
    std::fprintf(f, "<?xml version=\"1.0\"?>\n");
    std::fprintf(f,
                 "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                 "byte_order=\"LittleEndian\">\n");
    std::fprintf(f, "  <UnstructuredGrid>\n");
    std::fprintf(f, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", npts, nc);

    std::fprintf(f,
                 "      <Points>\n        <DataArray type=\"Float64\" "
                 "NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < npts; ++i)
        std::fprintf(f, "          %.17g %.17g %.17g\n", mp.node_xyz[3 * i], mp.node_xyz[3 * i + 1],
                     mp.node_xyz[3 * i + 2]);
    std::fprintf(f, "        </DataArray>\n      </Points>\n");

    std::fprintf(f,
                 "      <Cells>\n        <DataArray type=\"Int32\" "
                 "Name=\"connectivity\" format=\"ascii\">\n");
    for (const OutCell& c : cells) {
        std::fprintf(f, "          ");
        for (int k = 0; k < c.nn; ++k) std::fprintf(f, "%d ", c.nodes[k]);
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "        </DataArray>\n");
    std::fprintf(f,
                 "        <DataArray type=\"Int32\" Name=\"offsets\" "
                 "format=\"ascii\">\n          ");
    int off = 0;
    for (const OutCell& c : cells) {
        off += c.nn;
        std::fprintf(f, "%d ", off);
    }
    std::fprintf(f, "\n        </DataArray>\n");
    std::fprintf(f,
                 "        <DataArray type=\"UInt8\" Name=\"types\" "
                 "format=\"ascii\">\n          ");
    for (const OutCell& c : cells)
        std::fprintf(f, "%d ", vtk_cell_type(static_cast<CellType>(c.type)));
    std::fprintf(f, "\n        </DataArray>\n      </Cells>\n");

    std::fprintf(f, "      <CellData>\n");
    std::fprintf(f,
                 "        <DataArray type=\"Int32\" Name=\"rank\" "
                 "format=\"ascii\">\n          ");
    for (int i = 0; i < nc; ++i) std::fprintf(f, "%d ", mp.rank);
    std::fprintf(f, "\n        </DataArray>\n");
    std::fprintf(f,
                 "        <DataArray type=\"Int32\" Name=\"%s\" "
                 "format=\"ascii\">\n          ",
                 data_name);
    for (const OutCell& c : cells) std::fprintf(f, "%d ", c.data);
    std::fprintf(f, "\n        </DataArray>\n");
    std::fprintf(f, "      </CellData>\n");

    std::fprintf(f, "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
    std::fclose(f);

    // --- .pvtu collection (rank 0) ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        char ppath[512];
        std::snprintf(ppath, sizeof ppath, "%s/%s.pvtu", outdir.c_str(), stem.c_str());
        FILE* pf = std::fopen(ppath, "w");
        if (!pf) {
            log_warn_rank("could not open %s", ppath);
            return;
        }
        std::fprintf(pf, "<?xml version=\"1.0\"?>\n");
        std::fprintf(pf,
                     "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
                     "byte_order=\"LittleEndian\">\n");
        std::fprintf(pf, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
        std::fprintf(pf, "    <PPointData>\n");
        std::fprintf(pf, "    </PPointData>\n");
        std::fprintf(pf, "    <PCellData>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"rank\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"%s\"/>\n", data_name);
        std::fprintf(pf, "    </PCellData>\n");
        std::fprintf(pf, "    <PPoints>\n");
        std::fprintf(pf,
                     "      <PDataArray type=\"Float64\" "
                     "NumberOfComponents=\"3\"/>\n");
        std::fprintf(pf, "    </PPoints>\n");
        for (int p = 0; p < nprocs; ++p)
            std::fprintf(pf, "    <Piece Source=\"%s_%05d.vtu\"/>\n", stem.c_str(), p);
        std::fprintf(pf, "  </PUnstructuredGrid>\n</VTKFile>\n");
        std::fclose(pf);
        log_stat("VTU: %s", ppath);
    }
}

}  // namespace

void write_vtu(const MeshPart& mp, const std::string& outdir, const std::string& stem) {
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);  // create the directory if needed

    // Two pieces per rank, each with strictly valid field values:
    //   <stem>_XXXXX.vtu     — owned volume cells only (CellData: rank,
    //                          global_id). Ghost cells are NOT exported: they
    //                          duplicate neighbouring blocks and were already
    //                          verified visually.
    //   <stem>_bnd_XXXXX.vtu — boundary faces only (CellData: rank, patch).
    std::vector<OutCell> volume;
    volume.reserve(mp.n_own);
    for (int i = 0; i < mp.n_own; ++i) {
        OutCell c{};
        c.type = mp.cell_type[i];
        c.nn = kNodesPerType[mp.cell_type[i]];
        for (int k = 0; k < c.nn; ++k) c.nodes[k] = mp.cell_nodes[i * 8 + k];
        c.data = mp.cell_gid[i];
        volume.push_back(c);
    }
    std::vector<OutCell> boundary;
    boundary.reserve(mp.n_faces);
    for (int i = 0; i < mp.n_faces; ++i) {
        if (mp.face_neigh[i] >= 0) continue;  // boundary faces only
        OutCell c{};
        c.type = mp.face_type[i];
        c.nn = (c.type == static_cast<uint8_t>(CellType::TRI)) ? 3 : 4;
        for (int k = 0; k < c.nn; ++k) c.nodes[k] = mp.face_nodes[i * 4 + k];
        c.data = mp.face_patch[i];
        boundary.push_back(c);
    }

    write_piece(mp, outdir, stem, volume, "global_id");
    write_piece(mp, outdir, stem + "_bnd", boundary, "patch");
}

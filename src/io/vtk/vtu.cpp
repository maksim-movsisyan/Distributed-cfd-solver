#include "cfd/io/vtk/vtu.hpp"

#include <mpi.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "cfd/mpi/log.hpp"

namespace {

const char* cell_type_name(uint8_t t) {
    switch (static_cast<CellType>(t)) {
        case CellType::TRI:
            return "Tri";
        case CellType::QUAD:
            return "Quad";
        case CellType::TET:
            return "Tetra";
        case CellType::PYRA:
            return "Pyramid";
        case CellType::PRISM:
            return "Wedge";
        case CellType::HEXA:
            return "Hexahedron";
    }
    return "?";
}

}  // namespace

void write_vtu(const MeshPart& mp, const std::string& outdir, const std::string& stem) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);  // create the directory if needed

    // Cells to write: owned (ghost=0), ghosts (ghost=1), boundary faces
    // (ghost=2, patch id in 'patch').
    struct OutCell {
        uint8_t type;  // CellType
        int32_t nodes[8];
        int nn;
        int32_t gid;
        int32_t patch;
        uint8_t ghost;
    };
    std::vector<OutCell> cells;
    cells.reserve(mp.n_cells + mp.n_faces);
    for (int i = 0; i < mp.n_cells; ++i) {
        OutCell c{};
        c.type = mp.cell_type[i];
        c.nn = kNodesPerType[mp.cell_type[i]];
        for (int k = 0; k < c.nn; ++k) c.nodes[k] = mp.cell_nodes[i * 8 + k];
        c.gid = mp.cell_gid[i];
        c.patch = -1;
        c.ghost = (i >= mp.n_own) ? 1 : 0;
        cells.push_back(c);
    }
    for (int i = 0; i < mp.n_faces; ++i) {
        if (mp.face_neigh[i] >= 0) continue;  // boundary faces only
        OutCell c{};
        c.type = mp.face_type[i];
        c.nn = (c.type == static_cast<uint8_t>(CellType::TRI)) ? 3 : 4;
        for (int k = 0; k < c.nn; ++k) c.nodes[k] = mp.face_nodes[i * 4 + k];
        c.gid = -1;
        c.patch = mp.face_patch[i];
        c.ghost = 2;
        cells.push_back(c);
    }
    const int nc = static_cast<int>(cells.size());

    // The point array is simply the local nodes.
    const int npts = mp.n_nodes;

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
    auto scalar_arr = [&](const char* type, const char* name, auto get) {
        std::fprintf(f,
                     "        <DataArray type=\"%s\" Name=\"%s\" "
                     "format=\"ascii\">\n          ",
                     type, name);
        for (const OutCell& c : cells) std::fprintf(f, "%d ", (int)get(c));
        std::fprintf(f, "\n        </DataArray>\n");
    };
    scalar_arr("Int32", "rank", [&](const OutCell&) { return mp.rank; });
    scalar_arr("UInt8", "ghost", [&](const OutCell& c) { return c.ghost; });
    scalar_arr("Int32", "global_id", [&](const OutCell& c) { return c.gid; });
    scalar_arr("Int32", "patch", [&](const OutCell& c) { return c.patch; });
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
        std::fprintf(pf, "  <PUnstructuredGrid GhostLevel=\"1\">\n");
        std::fprintf(pf, "    <PPointData>\n");
        std::fprintf(pf, "    </PPointData>\n");
        std::fprintf(pf, "    <PCellData>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"rank\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"UInt8\" Name=\"ghost\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"global_id\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"patch\"/>\n");
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

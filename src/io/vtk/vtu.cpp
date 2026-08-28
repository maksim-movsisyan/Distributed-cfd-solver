#include "cfd/io/vtk/vtu.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::io::vtk {

namespace {

constexpr LocalIndex kInvalidLocal = static_cast<LocalIndex>(-1);

// Maps internal CellType to VTK Unstructured Grid cell type ID
[[nodiscard]] inline uint8_t to_vtk_cell_type(mesh::CellType type) noexcept {
    switch (type) {
        case mesh::CellType::TRI:   return 5;  // VTK_TRIANGLE
        case mesh::CellType::QUAD:  return 9;  // VTK_QUAD
        case mesh::CellType::TET:   return 10; // VTK_TETRA
        case mesh::CellType::HEXA:  return 12; // VTK_HEXAHEDRON
        case mesh::CellType::PRISM: return 13; // VTK_WEDGE
        case mesh::CellType::PYRA:  return 14; // VTK_PYRAMID
        default:                    return 0;  // VTK_EMPTY_CELL
    }
}

// Shared <Points> + <Cells> sections of a volume piece (owned cells).
void write_points_and_cells(FILE* f, const mesh::MeshPart& mp) {
    const int n_cells = static_cast<int>(mp.n_own);
    const int n_pts   = static_cast<int>(mp.n_nodes);

    // Points Coordinates (SoA -> 3D components)
    std::fprintf(f, "      <Points>\n");
    std::fprintf(f, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < n_pts; ++i) {
        const auto i_sz = static_cast<std::size_t>(i);
        std::fprintf(f, "          %.16g %.16g %.16g\n", mp.node_x[i_sz], mp.node_y[i_sz], mp.node_z[i_sz]);
    }
    std::fprintf(f, "        </DataArray>\n");
    std::fprintf(f, "      </Points>\n");

    // Cells Section
    std::fprintf(f, "      <Cells>\n");

    // Connectivity via CSR offsets (Zero extra allocations)
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (int c = 0; c < n_cells; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        const LocalIndex start = mp.cell_nodes_offsets[c_sz];
        const LocalIndex end   = mp.cell_nodes_offsets[c_sz + 1];

        std::fprintf(f, "          ");
        for (LocalIndex k = start; k < end; ++k) {
            std::fprintf(f, "%d ", mp.cell_nodes[static_cast<std::size_t>(k)]);
        }
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "        </DataArray>\n");

    // Offsets
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        std::fprintf(f, "%d ", mp.cell_nodes_offsets[c_sz + 1]);
    }
    std::fprintf(f, "\n        </DataArray>\n");

    // Types
    std::fprintf(f, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        std::fprintf(f, "%u ", to_vtk_cell_type(mp.cell_type[c_sz]));
    }
    std::fprintf(f, "\n        </DataArray>\n");
    std::fprintf(f, "      </Cells>\n");
}

// =============================================================================
// 1. Volume Piece Writer (Owned Cells [0, n_own))
// =============================================================================

void write_volume_vtu(
    const mesh::MeshPart& mp,
    const std::string& outdir,
    const std::string& stem) {
    const int rank = mp.rank;
    const int n_cells = static_cast<int>(mp.n_own);
    const int n_pts   = static_cast<int>(mp.n_nodes);

    char filepath[512];
    std::snprintf(filepath, sizeof(filepath), "%s/%s_%05d.vtu", outdir.c_str(), stem.c_str(), rank);

    FILE* f = std::fopen(filepath, "wb");
    if (!f) {
        mpi::log_stat("WARNING: Could not open VTU file: %s", filepath);
        return;
    }

    // 1MB I/O buffer to minimize POSIX write syscalls
    std::vector<char> io_buffer(1024 * 1024);
    std::setvbuf(f, io_buffer.data(), _IOFBF, io_buffer.size());

    std::fprintf(f, "<?xml version=\"1.0\"?>\n");
    std::fprintf(f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(f, "  <UnstructuredGrid>\n");
    std::fprintf(f, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", n_pts, n_cells);

    write_points_and_cells(f, mp);

    // CellData Section
    std::fprintf(f, "      <CellData>\n");

    // Field: rank
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"rank\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) std::fprintf(f, "%d ", rank);
    std::fprintf(f, "\n        </DataArray>\n");

    // Field: global_id
    std::fprintf(f, "        <DataArray type=\"Int64\" Name=\"global_id\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) {
        std::fprintf(f, "%lld ", static_cast<long long>(mp.cell_gid[static_cast<std::size_t>(c)]));
    }
    std::fprintf(f, "\n        </DataArray>\n");

    // Field: local_id
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"local_id\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) std::fprintf(f, "%d ", c);
    std::fprintf(f, "\n        </DataArray>\n");

    // Field: volume
    std::fprintf(f, "        <DataArray type=\"Float64\" Name=\"volume\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) {
        std::fprintf(f, "%.10e ", mp.cell_volume[static_cast<std::size_t>(c)]);
    }
    std::fprintf(f, "\n        </DataArray>\n");

    std::fprintf(f, "      </CellData>\n");
    std::fprintf(f, "    </Piece>\n");
    std::fprintf(f, "  </UnstructuredGrid>\n");
    std::fprintf(f, "</VTKFile>\n");

    std::fclose(f);
}

// =============================================================================
// 2. Boundary Piece Writer (Boundary Faces Only)
// =============================================================================

void write_boundary_vtu(
    const mesh::MeshPart& mp,
    const std::string& outdir,
    const std::string& stem) {
    const int rank = mp.rank;
    const int n_pts = static_cast<int>(mp.n_nodes);

    // Count boundary faces
    std::vector<LocalIndex> bface_indices;
    bface_indices.reserve(static_cast<std::size_t>(mp.n_faces) / 4);

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        if (mp.face_neigh[static_cast<std::size_t>(f)] == kInvalidLocal) {
            bface_indices.push_back(f);
        }
    }

    const int n_bfaces = static_cast<int>(bface_indices.size());

    char filepath[512];
    std::snprintf(filepath, sizeof(filepath), "%s/%s_bnd_%05d.vtu", outdir.c_str(), stem.c_str(), rank);

    FILE* f = std::fopen(filepath, "wb");
    if (!f) {
        mpi::log_stat("WARNING: Could not open boundary VTU file: %s", filepath);
        return;
    }

    std::vector<char> io_buffer(1024 * 1024);
    std::setvbuf(f, io_buffer.data(), _IOFBF, io_buffer.size());

    std::fprintf(f, "<?xml version=\"1.0\"?>\n");
    std::fprintf(f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(f, "  <UnstructuredGrid>\n");
    std::fprintf(f, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", n_pts, n_bfaces);

    // Points
    std::fprintf(f, "      <Points>\n");
    std::fprintf(f, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < n_pts; ++i) {
        const auto i_sz = static_cast<std::size_t>(i);
        std::fprintf(f, "          %.16g %.16g %.16g\n", mp.node_x[i_sz], mp.node_y[i_sz], mp.node_z[i_sz]);
    }
    std::fprintf(f, "        </DataArray>\n");
    std::fprintf(f, "      </Points>\n");

    // Cells (Boundary Faces)
    std::fprintf(f, "      <Cells>\n");

    // Connectivity
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (LocalIndex bf : bface_indices) {
        const auto bf_sz = static_cast<std::size_t>(bf);
        const LocalIndex start = mp.face_nodes_offsets[bf_sz];
        const LocalIndex end   = mp.face_nodes_offsets[bf_sz + 1];

        std::fprintf(f, "          ");
        for (LocalIndex k = start; k < end; ++k) {
            std::fprintf(f, "%d ", mp.face_nodes[static_cast<std::size_t>(k)]);
        }
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "        </DataArray>\n");

    // Offsets
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ");
    int running_off = 0;
    for (LocalIndex bf : bface_indices) {
        const auto bf_sz = static_cast<std::size_t>(bf);
        running_off += static_cast<int>(mp.face_nodes_offsets[bf_sz + 1] - mp.face_nodes_offsets[bf_sz]);
        std::fprintf(f, "%d ", running_off);
    }
    std::fprintf(f, "\n        </DataArray>\n");

    // Types
    std::fprintf(f, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ");
    for (LocalIndex bf : bface_indices) {
        const auto bf_sz = static_cast<std::size_t>(bf);
        std::fprintf(f, "%u ", to_vtk_cell_type(mp.face_type[bf_sz]));
    }
    std::fprintf(f, "\n        </DataArray>\n");
    std::fprintf(f, "      </Cells>\n");

    // CellData Section
    std::fprintf(f, "      <CellData>\n");

    // Field: rank
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"rank\" format=\"ascii\">\n          ");
    for (int i = 0; i < n_bfaces; ++i) std::fprintf(f, "%d ", rank);
    std::fprintf(f, "\n        </DataArray>\n");

    // Field: patch_id
    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"patch_id\" format=\"ascii\">\n          ");
    for (LocalIndex bf : bface_indices) {
        std::fprintf(f, "%d ", mp.face_patch[static_cast<std::size_t>(bf)]);
    }
    std::fprintf(f, "\n        </DataArray>\n");

    // Field: area
    std::fprintf(f, "        <DataArray type=\"Float64\" Name=\"area\" format=\"ascii\">\n          ");
    for (LocalIndex bf : bface_indices) {
        std::fprintf(f, "%.10e ", mp.face_area[static_cast<std::size_t>(bf)]);
    }
    std::fprintf(f, "\n        </DataArray>\n");

    std::fprintf(f, "      </CellData>\n");
    std::fprintf(f, "    </Piece>\n");
    std::fprintf(f, "  </UnstructuredGrid>\n");
    std::fprintf(f, "</VTKFile>\n");

    std::fclose(f);
}

// =============================================================================
// 3. Parallel Master PVTU Writers (Rank 0)
// =============================================================================

void write_pvtu_master(
    const std::string& outdir,
    const std::string& filename,
    const std::string& piece_prefix,
    int nprocs,
    bool is_boundary) {
    char ppath[512];
    std::snprintf(ppath, sizeof(ppath), "%s/%s.pvtu", outdir.c_str(), filename.c_str());

    FILE* pf = std::fopen(ppath, "wb");
    if (!pf) {
        mpi::log_stat("WARNING: Could not open PVTU file: %s", ppath);
        return;
    }

    std::fprintf(pf, "<?xml version=\"1.0\"?>\n");
    std::fprintf(pf, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(pf, "  <PUnstructuredGrid GhostLevel=\"0\">\n");

    // PCellData Header
    std::fprintf(pf, "    <PCellData>\n");
    std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"rank\"/>\n");
    if (!is_boundary) {
        std::fprintf(pf, "      <PDataArray type=\"Int64\" Name=\"global_id\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"local_id\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"volume\"/>\n");
    } else {
        std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"patch_id\"/>\n");
        std::fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"area\"/>\n");
    }
    std::fprintf(pf, "    </PCellData>\n");

    // PPoints Header
    std::fprintf(pf, "    <PPoints>\n");
    std::fprintf(pf, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
    std::fprintf(pf, "    </PPoints>\n");

    // Piece References
    for (int p = 0; p < nprocs; ++p) {
        std::fprintf(pf, "    <Piece Source=\"%s_%05d.vtu\"/>\n", piece_prefix.c_str(), p);
    }

    std::fprintf(pf, "  </PUnstructuredGrid>\n");
    std::fprintf(pf, "</VTKFile>\n");

    std::fclose(pf);
    mpi::log_stat("INFO: Visualisation file written: %s", ppath);
}

// Solution volume piece: geometry + rank + the requested solution fields.
void write_solution_vtu_piece(
    const mesh::MeshPart& mp,
    const SolutionField* fields,
    int nfields,
    const std::string& outdir,
    const std::string& stem) {
    const int rank = mp.rank;
    const int n_cells = static_cast<int>(mp.n_own);
    const int n_pts   = static_cast<int>(mp.n_nodes);

    char filepath[512];
    std::snprintf(filepath, sizeof(filepath), "%s/%s_%05d.vtu", outdir.c_str(), stem.c_str(), rank);

    FILE* f = std::fopen(filepath, "wb");
    if (!f) {
        mpi::log_stat("WARNING: Could not open VTU file: %s", filepath);
        return;
    }

    std::vector<char> io_buffer(1024 * 1024);
    std::setvbuf(f, io_buffer.data(), _IOFBF, io_buffer.size());

    std::fprintf(f, "<?xml version=\"1.0\"?>\n");
    std::fprintf(f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(f, "  <UnstructuredGrid>\n");
    std::fprintf(f, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", n_pts, n_cells);

    write_points_and_cells(f, mp);

    std::fprintf(f, "      <CellData>\n");

    std::fprintf(f, "        <DataArray type=\"Int32\" Name=\"rank\" format=\"ascii\">\n          ");
    for (int c = 0; c < n_cells; ++c) std::fprintf(f, "%d ", rank);
    std::fprintf(f, "\n        </DataArray>\n");

    for (int i = 0; i < nfields; ++i) {
        std::fprintf(f, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n          ",
                     fields[i].name);
        for (int c = 0; c < n_cells; ++c) {
            std::fprintf(f, "%.10e ", fields[i].values[static_cast<std::size_t>(c)]);
        }
        std::fprintf(f, "\n        </DataArray>\n");
    }

    std::fprintf(f, "      </CellData>\n");
    std::fprintf(f, "    </Piece>\n");
    std::fprintf(f, "  </UnstructuredGrid>\n");
    std::fprintf(f, "</VTKFile>\n");

    std::fclose(f);
}

void write_solution_pvtu_master(
    const std::string& outdir,
    const std::string& stem,
    int nprocs,
    const SolutionField* fields,
    int nfields) {
    char ppath[512];
    std::snprintf(ppath, sizeof(ppath), "%s/%s.pvtu", outdir.c_str(), stem.c_str());

    FILE* pf = std::fopen(ppath, "wb");
    if (!pf) {
        mpi::log_stat("WARNING: Could not open PVTU file: %s", ppath);
        return;
    }

    std::fprintf(pf, "<?xml version=\"1.0\"?>\n");
    std::fprintf(pf, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(pf, "  <PUnstructuredGrid GhostLevel=\"0\">\n");

    std::fprintf(pf, "    <PCellData>\n");
    std::fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"rank\"/>\n");
    for (int i = 0; i < nfields; ++i) {
        std::fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"%s\"/>\n", fields[i].name);
    }
    std::fprintf(pf, "    </PCellData>\n");

    std::fprintf(pf, "    <PPoints>\n");
    std::fprintf(pf, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
    std::fprintf(pf, "    </PPoints>\n");

    for (int p = 0; p < nprocs; ++p) {
        std::fprintf(pf, "    <Piece Source=\"%s_%05d.vtu\"/>\n", stem.c_str(), p);
    }

    std::fprintf(pf, "  </PUnstructuredGrid>\n");
    std::fprintf(pf, "</VTKFile>\n");

    std::fclose(pf);
    mpi::log_stat("INFO: Solution file written: %s", ppath);
}

} // anonymous namespace

// =============================================================================
// Public write_vtu Entry Point
// =============================================================================

void write_vtu(
    const mesh::MeshPart& mp,
    const std::string& outdir,
    const std::string& stem, 
    MPI_Comm comm) {
    const int rank = mp.rank;
    const int nprocs = mp.nprocs;

    // Create target directory
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
    }
    MPI_Barrier(comm);

    // 1. Parallel write of rank pieces
    write_volume_vtu(mp, outdir, stem);
    write_boundary_vtu(mp, outdir, stem);

    MPI_Barrier(comm);

    // 2. Rank 0 writes master .pvtu collections
    if (rank == 0) {
        write_pvtu_master(outdir, stem, stem, nprocs, false);
        write_pvtu_master(outdir, stem + "_bnd", stem + "_bnd", nprocs, true);
    }
}

// =============================================================================
// Public write_solution_vtu Entry Point
// =============================================================================

void write_solution_vtu(
    const mesh::MeshPart& mp,
    const SolutionField* fields,
    int nfields,
    const std::string& outdir,
    const std::string& stem,
    MPI_Comm comm) {
    const int rank = mp.rank;

    if (rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
    }
    MPI_Barrier(comm);

    write_solution_vtu_piece(mp, fields, nfields, outdir, stem);

    MPI_Barrier(comm);

    if (rank == 0) {
        write_solution_pvtu_master(outdir, stem, mp.nprocs, fields, nfields);
    }
}

} // namespace cfd::io::vtk
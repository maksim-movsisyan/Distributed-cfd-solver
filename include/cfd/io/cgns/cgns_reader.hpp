// Parallel CGNS reading via PCGNS (cgp_* on top of parallel HDF5).
//
// Supported: 1 base, 1 zone (Unstructured), per-type element sections without
// MIXED: TETRA_4, PYRA_5, PENTA_6, HEXA_8 (volume) + TRI_3, QUAD_4 (boundary,
// for BCs). BAR_* are skipped. BCs: ZoneBC with PointList/PointRange (GridLocation
// = FaceCenter); the name is the FamilyName when present, else the BC node name.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <cgnslib.h>
#include <pcgnslib.h>
#include <mpi.h>

#include "cfd/mesh/raw_mesh.hpp"

namespace cfd::io::cgns {

using DataType = CGNS_ENUMT(DataType_t);                // e.g Integer / LongInteger / RealSingle ... 
using ZoneType = CGNS_ENUMT(ZoneType_t);                // e.g Structured / Unstructured 
using ElementType = CGNS_ENUMT(ElementType_t);          // e.g TETRA_4 / PYRA_5 / PENTA_6 / HEXA_8 / ... | TRI_3 / QUAD_4 / MIXED
using BoundaryConditionType = CGNS_ENUMT(BCType_t);     // e.g BCOutflowSubsonic / BCInflowSubsonic / BCWallInviscid ...
using PointSetType = CGNS_ENUMT(PointSetType_t);        // e.g PointRange / PointList
using GridLocation = CGNS_ENUMT(GridLocation_t);        // e.g Vertex / CellCenter / FaceCenter

using PatchId = std::int32_t;
inline constexpr PatchId kInvalidPatchId = -1;



inline void check(int status, std::string_view operation) {
    if (status == CG_OK) { return; }

    std::string message{operation};
    const char* const cgns_message = cg_get_error();

    if (cgns_message != nullptr && cgns_message[0] != '\0') {
        message += ": ";
        message += cgns_message;
    }

    throw std::runtime_error(message);
}


class File final {
public:
    explicit File(std::string path, MPI_Comm comm = MPI_COMM_WORLD) 
        : path_(std::move(path)), comm_(comm) {
        if (path_.empty()) {
            throw std::invalid_argument{"CGNS file path is empty"};
        }

        // Configure communicator for parallel HDF5/PCGNS
        check(cgp_mpi_comm(comm_), "cgp_mpi_comm");
        check(cgp_open(path_.c_str(), CG_MODE_READ, &id_), "cgp_open");
    }

    ~File() noexcept { if (is_open()) static_cast<void>(cgp_close(id_)); }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

    File(File&& o) noexcept 
        : path_(std::move(o.path_)), id_(o.id_), comm_(o.comm_) {
        o.id_ = invalid_id;
    }

    File& operator=(File&& o) noexcept {
        if (this != &o) {
             if (is_open()) static_cast<void>(cgp_close(id_));
            path_ = std::move(o.path_);
            id_ = o.id_;
            comm_ = o.comm_;
            o.id_ = invalid_id;
        }
        return *this;
    }

    [[nodiscard]] int id() const noexcept { return id_; }
    [[nodiscard]] MPI_Comm comm() const noexcept { return comm_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] bool is_open() const noexcept { return id_ != invalid_id; }

    void close() {
        if (!is_open()) { return; }

        const int id_to_close = id_;
        id_ = invalid_id;

        // close file
        check(cgp_close(id_to_close), "cgp_close");
    }

private:
    static constexpr int invalid_id = -1;

    std::string path_;
    int id_{invalid_id};
    MPI_Comm comm_{MPI_COMM_NULL};
};

// Parallel read (all ranks of the communicator). On file incompatibility it
// prints an error and returns nullptr on every rank.
mesh::RawMesh read_cgns_parallel(const std::string& path, MPI_Comm comm = MPI_COMM_WORLD);

// Fetch node coordinates by global ids (the node owner answers from its own
// slice). Returns interleaved xyz triples (size = 3 * node_gids.size()) 
// in input-list order (duplicates allowed).
//std::vector<double> fetch_coords(RawMesh& m, const std::vector<GlobalIndex>& node_gids);

} //namespace cfd::io::cgns 
#include "cfd/io/solver_mesh/hdf5_writer.hpp"

#include <hdf5.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>

#include "cfd/mpi/log.hpp"

namespace cfd::io::solver_mesh {

namespace {

// =============================================================================
// Compile-Time Type Resolver: C++ Type -> HDF5 Native Datatype
// =============================================================================

template <typename T>
[[nodiscard]] inline hid_t get_h5_type() noexcept {
    if constexpr (std::is_same_v<T, float>)                     return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>)               return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, int8_t>)               return H5T_NATIVE_INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)              return H5T_NATIVE_UINT8;
    else if constexpr (std::is_same_v<T, int16_t>)              return H5T_NATIVE_INT16;
    else if constexpr (std::is_same_v<T, uint16_t>)             return H5T_NATIVE_UINT16;
    else if constexpr (std::is_same_v<T, int32_t>)              return H5T_NATIVE_INT32;
    else if constexpr (std::is_same_v<T, uint32_t>)             return H5T_NATIVE_UINT32;
    else if constexpr (std::is_same_v<T, int64_t>)              return H5T_NATIVE_INT64;
    else if constexpr (std::is_same_v<T, uint64_t>)             return H5T_NATIVE_UINT64;
    else if constexpr (sizeof(T) == 4 && std::is_integral_v<T>) return H5T_NATIVE_INT32;
    else if constexpr (sizeof(T) == 8 && std::is_integral_v<T>) return H5T_NATIVE_INT64;
    else return H5I_INVALID_HID;
}

// =============================================================================
// RAII Guard for HDF5 Identifiers
// =============================================================================

struct H5Id {
    hid_t id = H5I_INVALID_HID;

    H5Id() noexcept = default;
    explicit H5Id(hid_t h) noexcept : id(h) {}
    ~H5Id() { reset(); }

    H5Id(const H5Id&) = delete;
    H5Id& operator=(const H5Id&) = delete;

    H5Id(H5Id&& o) noexcept : id(o.id) { o.id = H5I_INVALID_HID; }
    H5Id& operator=(H5Id&& o) noexcept {
        if (this != &o) {
            reset();
            id = o.id;
            o.id = H5I_INVALID_HID;
        }
        return *this;
    }

    [[nodiscard]] operator hid_t() const noexcept { return id; }

    void reset() noexcept {
        if (id > 0) {
            H5I_type_t type = H5Iget_type(id);
            if (type == H5I_FILE)            H5Fclose(id);
            else if (type == H5I_GROUP)      H5Gclose(id);
            else if (type == H5I_DATASET)    H5Dclose(id);
            else if (type == H5I_DATASPACE)  H5Sclose(id);
            else if (type == H5I_DATATYPE)   H5Tclose(id);
            else if (type == H5I_GENPROP_LST) H5Pclose(id);
            else if (type == H5I_ATTR)       H5Aclose(id);
            id = H5I_INVALID_HID;
        }
    }
};

// =============================================================================
// Parallel Topology-Aware Offset Calculation
// =============================================================================

struct Hyperslab1D {
    hsize_t global_size = 0;
    hsize_t local_offset = 0;
    hsize_t local_count = 0;
    std::vector<uint64_t> part_offsets; // Size: nprocs + 1
};

Hyperslab1D compute_topology_offsets(uint64_t local_count, int part_id, int nprocs, MPI_Comm comm) {
    const auto nprocs_sz = static_cast<std::size_t>(nprocs);

    struct RankPartCount {
        int part_id;
        uint64_t count;
    } my_data{part_id, local_count};

    std::vector<RankPartCount> all_data(nprocs_sz);
    MPI_Allgather(&my_data, sizeof(RankPartCount), MPI_BYTE,
                  all_data.data(), sizeof(RankPartCount), MPI_BYTE, comm);

    Hyperslab1D layout;
    layout.local_count = static_cast<hsize_t>(local_count);
    layout.part_offsets.assign(nprocs_sz + 1, 0);

    std::vector<uint64_t> counts_by_part(nprocs_sz, 0);
    for (const auto& item : all_data) {
        counts_by_part[static_cast<std::size_t>(item.part_id)] = item.count;
    }

    for (std::size_t p = 0; p < nprocs_sz; ++p) {
        layout.part_offsets[p + 1] = layout.part_offsets[p] + counts_by_part[p];
    }

    layout.global_size = layout.part_offsets.back();
    layout.local_offset = layout.part_offsets[static_cast<std::size_t>(part_id)];

    return layout;
}

// =============================================================================
// Parallel Collective Dataset Writers (Type-Safe)
// =============================================================================

template <typename T>
void write_dataset_1d(
    hid_t loc_id,
    const std::string& name,
    const Hyperslab1D& layout,
    const T* data,
    hid_t dxpl_id)
{
    hid_t h5_type = get_h5_type<T>();

    hsize_t dims[1] = {layout.global_size};
    H5Id filespace(H5Screate_simple(1, dims, nullptr));
    H5Id dset(H5Dcreate2(loc_id, name.c_str(), h5_type, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    H5Id memspace;
    if (layout.local_count > 0) {
        hsize_t count[1] = {layout.local_count};
        hsize_t offset[1] = {layout.local_offset};
        memspace = H5Id(H5Screate_simple(1, count, nullptr));
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    } else {
        memspace = H5Id(H5Screate(H5S_NULL));
        H5Sselect_none(filespace);
    }

    H5Dwrite(dset, h5_type, memspace, filespace, dxpl_id, data);
}

void write_dataset_2d_vec3(
    hid_t loc_id,
    const std::string& name,
    const Hyperslab1D& layout,
    const double* x,
    const double* y,
    const double* z,
    hid_t dxpl_id)
{
    hsize_t dims[2] = {layout.global_size, 3};
    H5Id filespace(H5Screate_simple(2, dims, nullptr));
    H5Id dset(H5Dcreate2(loc_id, name.c_str(), H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    std::vector<double> flat_buf(static_cast<std::size_t>(layout.local_count) * 3);
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.local_count); ++i) {
        flat_buf[i * 3 + 0] = x[i];
        flat_buf[i * 3 + 1] = y[i];
        flat_buf[i * 3 + 2] = z[i];
    }

    H5Id memspace;
    if (layout.local_count > 0) {
        hsize_t count[2] = {layout.local_count, 3};
        hsize_t offset[2] = {layout.local_offset, 0};
        memspace = H5Id(H5Screate_simple(2, count, nullptr));
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    } else {
        memspace = H5Id(H5Screate(H5S_NULL));
        H5Sselect_none(filespace);
    }

    H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_id, flat_buf.data());
}

} // anonymous namespace

// =============================================================================
// export_mesh_hdf5 Implementation
// =============================================================================

void export_mesh_hdf5(
    const mesh::MeshPart& mp,
    const partition::PartitionResult& pr,
    const std::string& filepath, MPI_Comm comm) {
    const int rank = mp.rank;
    const int nprocs = mp.nprocs;
    const int part_id = static_cast<int>(pr.rank2part[static_cast<std::size_t>(rank)]);

    // 1. Setup Parallel HDF5 File Access Properties
    H5Id fapl(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    H5Id file(H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
    if (file < 0) {
        std::stringstream ss;
        ss << "Failed to create Parallel HDF5 mesh file: " << filepath.c_str();
        auto result = ss.str();
        mpi::fatal(comm, result);
    }

    H5Id dxpl(H5Pcreate(H5P_DATASET_XFER));
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    // 2. Compute Topology-Aware Partition Offsets
    const auto cell_layout       = compute_topology_offsets(static_cast<uint64_t>(mp.n_cells), part_id, nprocs, comm);
    const auto cell_nodes_layout = compute_topology_offsets(static_cast<uint64_t>(mp.cell_nodes.size()), part_id, nprocs, comm);
    const auto face_layout       = compute_topology_offsets(static_cast<uint64_t>(mp.n_faces), part_id, nprocs, comm);
    const auto face_nodes_layout = compute_topology_offsets(static_cast<uint64_t>(mp.face_nodes.size()), part_id, nprocs, comm);
    const auto node_layout       = compute_topology_offsets(static_cast<uint64_t>(mp.n_nodes), part_id, nprocs, comm);

    const auto comm_nb_layout    = compute_topology_offsets(static_cast<uint64_t>(mp.nb_ranks.size()), part_id, nprocs, comm);
    const auto send_cell_layout  = compute_topology_offsets(static_cast<uint64_t>(mp.send_owned_local.size()), part_id, nprocs, comm);
    const auto recv_cell_layout  = compute_topology_offsets(static_cast<uint64_t>(mp.recv_ghost_local.size()), part_id, nprocs, comm);
    const auto patch_face_layout = compute_topology_offsets(static_cast<uint64_t>(mp.patch_faces.size()), part_id, nprocs, comm);

    // 3. Root Group Metadata Attributes
    {
        H5Id scalar_space(H5Screate(H5S_SCALAR));
        auto write_attr = [&](const char* name, hid_t type, const void* val) {
            H5Id attr(H5Acreate2(file, name, type, scalar_space, H5P_DEFAULT, H5P_DEFAULT));
            H5Awrite(attr, type, val);
        };

        const int nprocs_val = nprocs;
        const uint64_t n_cells_g_val  = static_cast<uint64_t>(mp.n_cells_g);
        const uint64_t n_faces_g_val  = static_cast<uint64_t>(mp.n_faces_g);
        const uint64_t n_bfaces_g_val = static_cast<uint64_t>(mp.n_bfaces_g);
        const uint64_t n_nodes_g_val  = static_cast<uint64_t>(mp.n_nodes_g);

        write_attr("nprocs", H5T_NATIVE_INT, &nprocs_val);
        write_attr("n_cells_global", H5T_NATIVE_UINT64, &n_cells_g_val);
        write_attr("n_faces_global", H5T_NATIVE_UINT64, &n_faces_g_val);
        write_attr("n_bfaces_global", H5T_NATIVE_UINT64, &n_bfaces_g_val);
        write_attr("n_nodes_global", H5T_NATIVE_UINT64, &n_nodes_g_val);

        hsize_t b_dims[1] = {3};
        H5Id vec_space(H5Screate_simple(1, b_dims, nullptr));
        H5Id a_lo(H5Acreate2(file, "bbox_lo", H5T_NATIVE_DOUBLE, vec_space, H5P_DEFAULT, H5P_DEFAULT));
        H5Id a_hi(H5Acreate2(file, "bbox_hi", H5T_NATIVE_DOUBLE, vec_space, H5P_DEFAULT, H5P_DEFAULT));
        H5Awrite(a_lo, H5T_NATIVE_DOUBLE, mp.bbox_lo);
        H5Awrite(a_hi, H5T_NATIVE_DOUBLE, mp.bbox_hi);
    }

    // 4. Group: /partition
    {
        H5Id g_part(H5Gcreate2(file, "partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        hsize_t p_dims[1] = {static_cast<hsize_t>(nprocs + 1)};
        H5Id p_space(H5Screate_simple(1, p_dims, nullptr));

        auto write_part_table = [&](const char* name, const std::vector<uint64_t>& offsets) {
            H5Id dset(H5Dcreate2(g_part, name, H5T_NATIVE_UINT64, p_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            H5Dwrite(dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, dxpl, offsets.data());
        };

        write_part_table("cell_offsets", cell_layout.part_offsets);
        write_part_table("cell_nodes_offsets", cell_nodes_layout.part_offsets);
        write_part_table("face_offsets", face_layout.part_offsets);
        write_part_table("face_nodes_offsets", face_nodes_layout.part_offsets);
        write_part_table("node_offsets", node_layout.part_offsets);
        write_part_table("comm_nb_offsets", comm_nb_layout.part_offsets);

        hsize_t r_dims[1] = {static_cast<hsize_t>(nprocs)};
        H5Id r_space(H5Screate_simple(1, r_dims, nullptr));
        H5Id d_p2r(H5Dcreate2(g_part, "part2rank", H5T_NATIVE_INT, r_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        H5Id d_r2p(H5Dcreate2(g_part, "rank2part", H5T_NATIVE_INT, r_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        H5Dwrite(d_p2r, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, pr.part2rank.data());
        H5Dwrite(d_r2p, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, pr.rank2part.data());
    }

    // 5. Group: /cells
    {
        H5Id g_cells(H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        std::vector<uint8_t> cell_types(mp.cell_type.size());
        for (std::size_t i = 0; i < mp.cell_type.size(); ++i) {
            cell_types[i] = static_cast<uint8_t>(mp.cell_type[i]);
        }

        write_dataset_1d(g_cells, "type", cell_layout, cell_types.data(), dxpl);
        write_dataset_1d(g_cells, "gid", cell_layout, mp.cell_gid.data(), dxpl);
        write_dataset_1d(g_cells, "donor", cell_layout, mp.cell_donor.data(), dxpl);
        write_dataset_1d(g_cells, "volume", cell_layout, mp.cell_volume.data(), dxpl);
        write_dataset_1d(g_cells, "nodes_offsets", cell_layout, mp.cell_nodes_offsets.data(), dxpl);
        write_dataset_1d(g_cells, "nodes", cell_nodes_layout, mp.cell_nodes.data(), dxpl);

        write_dataset_2d_vec3(g_cells, "centroid", cell_layout,
                              mp.cell_centroid_x.data(), mp.cell_centroid_y.data(), mp.cell_centroid_z.data(), dxpl);
    }

    // 6. Group: /faces
    {
        H5Id g_faces(H5Gcreate2(file, "faces", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        std::vector<uint8_t> face_types(mp.face_type.size());
        for (std::size_t i = 0; i < mp.face_type.size(); ++i) {
            face_types[i] = static_cast<uint8_t>(mp.face_type[i]);
        }

        write_dataset_1d(g_faces, "owner", face_layout, mp.face_owner.data(), dxpl);
        write_dataset_1d(g_faces, "neigh", face_layout, mp.face_neigh.data(), dxpl);
        write_dataset_1d(g_faces, "patch", face_layout, mp.face_patch.data(), dxpl);
        write_dataset_1d(g_faces, "type", face_layout, face_types.data(), dxpl);
        write_dataset_1d(g_faces, "area", face_layout, mp.face_area.data(), dxpl);
        write_dataset_1d(g_faces, "nodes_offsets", face_layout, mp.face_nodes_offsets.data(), dxpl);
        write_dataset_1d(g_faces, "nodes", face_nodes_layout, mp.face_nodes.data(), dxpl);

        write_dataset_2d_vec3(g_faces, "normal", face_layout,
                              mp.face_normal_x.data(), mp.face_normal_y.data(), mp.face_normal_z.data(), dxpl);
        write_dataset_2d_vec3(g_faces, "centroid", face_layout,
                              mp.face_centroid_x.data(), mp.face_centroid_y.data(), mp.face_centroid_z.data(), dxpl);
    }

    // 7. Group: /nodes
    {
        H5Id g_nodes(H5Gcreate2(file, "nodes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        write_dataset_1d(g_nodes, "gid", node_layout, mp.node_gid.data(), dxpl);
        write_dataset_2d_vec3(g_nodes, "coords", node_layout,
                              mp.node_x.data(), mp.node_y.data(), mp.node_z.data(), dxpl);
    }

    // 8. Group: /comm
    {
        H5Id g_comm(H5Gcreate2(file, "comm", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        write_dataset_1d(g_comm, "nb_ranks", comm_nb_layout, mp.nb_ranks.data(), dxpl);
        write_dataset_1d(g_comm, "send_owned_cells", send_cell_layout, mp.send_owned_local.data(), dxpl);
        write_dataset_1d(g_comm, "recv_ghost_cells", recv_cell_layout, mp.recv_ghost_local.data(), dxpl);

        const auto comm_off_layout = compute_topology_offsets(
            static_cast<uint64_t>(mp.send_offsets.size()), part_id, nprocs, comm);
        write_dataset_1d(g_comm, "send_offsets", comm_off_layout, mp.send_offsets.data(), dxpl);
        write_dataset_1d(g_comm, "recv_offsets", comm_off_layout, mp.recv_offsets.data(), dxpl);
    }

    // 9. Group: /patches
    {
        H5Id g_patch(H5Gcreate2(file, "patches", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        const auto n_patches = mp.patches.size();

        hsize_t p_dims[1] = {static_cast<hsize_t>(n_patches)};
        H5Id p_space(H5Screate_simple(1, p_dims, nullptr));

        constexpr std::size_t kStrLen = 64;
        H5Id str_type(H5Tcopy(H5T_C_S1));
        H5Tset_size(str_type, kStrLen);

        std::vector<std::array<char, kStrLen>> names_buf(n_patches);
        std::vector<std::array<char, kStrLen>> types_buf(n_patches);

        for (std::size_t i = 0; i < n_patches; ++i) {
            std::strncpy(names_buf[i].data(), mp.patches[i].name.c_str(), kStrLen - 1);
            std::strncpy(types_buf[i].data(), mp.patches[i].cgns_type.c_str(), kStrLen - 1);
            names_buf[i][kStrLen - 1] = '\0';
            types_buf[i][kStrLen - 1] = '\0';
        }

        H5Id d_names(H5Dcreate2(g_patch, "names", str_type, p_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        H5Id d_types(H5Dcreate2(g_patch, "cgns_types", str_type, p_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        H5Dwrite(d_names, str_type, H5S_ALL, H5S_ALL, dxpl, names_buf.data());
        H5Dwrite(d_types, str_type, H5S_ALL, H5S_ALL, dxpl, types_buf.data());

        const auto patch_off_layout = compute_topology_offsets(
            static_cast<uint64_t>(mp.patch_face_offsets.size()), part_id, nprocs, comm);

        write_dataset_1d(g_patch, "patch_face_offsets", patch_off_layout, mp.patch_face_offsets.data(), dxpl);
        write_dataset_1d(g_patch, "patch_faces", patch_face_layout, mp.patch_faces.data(), dxpl);
    }

    if (rank == 0) {
        mpi::log_stat("INFO: Mesh successfully exported to Parallel HDF5: %s", filepath.c_str());
    }
}

} // namespace cfd::mesh
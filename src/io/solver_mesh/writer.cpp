#include "cfd/io/solver_mesh/writer.hpp"

#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cfd/io/solver_mesh/h5util.hpp"
#include "cfd/mpi/log.hpp"

// File format (v2, flat):
//   global attributes + per-rank offset datasets + global arrays
//   cells/*, nodes/*, faces/*, comm/*, patches/face_idx.
// The data of rank r occupies [off[r], off[r+1]) of every array.
// Writing: phase 1 — rank 0 creates the fully allocated skeleton (zeros);
// phase 2 — all ranks simultaneously write their hyperslabs with
// collective MPI-IO (the canonical parallel HDF5 pattern).

namespace {

long long my_start(long long local) {
    long long s = local;
    MPI_Exscan(MPI_IN_PLACE, &s, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) s = 0;  // Exscan is undefined on rank 0 per the standard
    return s;
}
long long all_total(long long local) {
    long long t = local;
    MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return t;
}

void write_i64_ds(hid_t file, const char* name, const std::vector<int64_t>& v) {
    h5_create_ds(file, name, H5T_NATIVE_INT64, {(hsize_t)v.size()});
    h5_write_existing(file, name, H5T_NATIVE_INT64, v.data(), {(hsize_t)v.size()});
}

std::vector<int64_t> gather_offsets(long long mine, int nprocs) {
    std::vector<long long> sz(nprocs);
    MPI_Gather(const_cast<long long*>(&mine), 1, MPI_LONG_LONG_INT, sz.data(), 1, MPI_LONG_LONG_INT,
               0, MPI_COMM_WORLD);
    std::vector<int64_t> off(nprocs + 1, 0);
    for (int r = 0; r < nprocs; ++r) off[r + 1] = off[r] + sz[r];
    return off;
}

}  // namespace

void write_mesh_file(const std::string& path, const MeshPart& mp, const GlobalMeta& gm) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // --- Local sizes and global offsets ---
    const long long ncells_l = mp.n_cells;
    const long long nnodes_l = mp.n_nodes;
    const long long nfaces_l = mp.n_faces;
    const long long nnb_l = mp.n_neighbors();
    const long long nrecv_l = (long long)mp.recv_ghost_local.size();
    const long long nsend_l = (long long)mp.send_owned_local.size();
    const long long npf_l = (long long)mp.patch_faces.size();
    const long long nseg_l = nnb_l + 1;  // segment boundaries (nnb+1)

    const long long cells_b = my_start(ncells_l), cells_g = all_total(ncells_l);
    const long long nodes_b = my_start(nnodes_l), nodes_g = all_total(nnodes_l);
    const long long faces_b = my_start(nfaces_l), faces_g = all_total(nfaces_l);
    const long long nb_b = my_start(nnb_l), nb_g = all_total(nnb_l);
    const long long recv_b = my_start(nrecv_l), recv_g = all_total(nrecv_l);
    const long long send_b = my_start(nsend_l), send_g = all_total(nsend_l);
    const long long pf_b = my_start(npf_l), pf_g = all_total(npf_l);
    const long long rseg_b = my_start(nseg_l), rseg_g = all_total(nseg_l);
    const long long sseg_b = my_start(nseg_l), sseg_g = all_total(nseg_l);

    // --- Phase 1: skeleton (rank 0 only, serial driver) ---
    // All gathers happen BEFORE the if: collectives must be called by
    // every rank.
    const std::vector<int64_t> cells_off = gather_offsets(ncells_l, nprocs);
    const std::vector<int64_t> nodes_off = gather_offsets(nnodes_l, nprocs);
    const std::vector<int64_t> faces_off = gather_offsets(nfaces_l, nprocs);
    const std::vector<int64_t> nb_off = gather_offsets(nnb_l, nprocs);
    const std::vector<int64_t> recv_off = gather_offsets(nrecv_l, nprocs);
    const std::vector<int64_t> send_off = gather_offsets(nsend_l, nprocs);
    const std::vector<int64_t> pf_off = gather_offsets(npf_l, nprocs);
    const std::vector<int64_t> rseg_off = gather_offsets(nseg_l, nprocs);
    const std::vector<int64_t> sseg_off = gather_offsets(nseg_l, nprocs);
    std::vector<int64_t> nown(nprocs);
    {
        const long long nown_l = mp.n_own;
        std::vector<long long> sz(nprocs);
        MPI_Gather(const_cast<long long*>(&nown_l), 1, MPI_LONG_LONG_INT, sz.data(), 1,
                   MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        for (int r = 0; r < nprocs; ++r) nown[r] = sz[r];
    }
    if (rank == 0) {
        H5Obj file = h5_create_file_serial(path);
        h5_attr(file, "format", H5T_NATIVE_INT32, std::vector<int32_t>{2}.data());
        h5_attr(file, "nprocs", H5T_NATIVE_INT32, std::vector<int32_t>{nprocs}.data());
        h5_attr(file, "n_cells_g", H5T_NATIVE_INT64, std::vector<int64_t>{gm.n_cells_g}.data());
        h5_attr(file, "n_faces_g", H5T_NATIVE_INT64, std::vector<int64_t>{gm.n_faces_g}.data());
        h5_attr(file, "n_nodes_g", H5T_NATIVE_INT64, std::vector<int64_t>{gm.n_nodes_g}.data());
        h5_attr(file, "n_bfaces_g", H5T_NATIVE_INT64, std::vector<int64_t>{gm.n_bfaces_g}.data());
        h5_attr(file, "total_volume", H5T_NATIVE_DOUBLE,
                std::vector<double>{gm.total_volume}.data());
        h5_attr(file, "n_flipped", H5T_NATIVE_INT64, std::vector<int64_t>{gm.n_flipped}.data());
        h5_attr(file, "n_ghost_total", H5T_NATIVE_INT64,
                std::vector<int64_t>{gm.n_ghost_total}.data());
        h5_attr_vec(file, "bbox_lo", H5T_NATIVE_DOUBLE, gm.bbox_lo, 3);
        h5_attr_vec(file, "bbox_hi", H5T_NATIVE_DOUBLE, gm.bbox_hi, 3);

        write_i64_ds(file, "cells_off", cells_off);
        write_i64_ds(file, "nodes_off", nodes_off);
        write_i64_ds(file, "faces_off", faces_off);
        write_i64_ds(file, "nb_off", nb_off);
        write_i64_ds(file, "recv_off", recv_off);
        write_i64_ds(file, "send_off", send_off);
        write_i64_ds(file, "patchfaces_off", pf_off);
        write_i64_ds(file, "recvseg_off", rseg_off);
        write_i64_ds(file, "sendseg_off", sseg_off);
        write_i64_ds(file, "n_own", nown);

        const size_t np = gm.patch_names.size();
        std::vector<char> names(np * 64, '\0'), types(np * 64, '\0');
        for (size_t p = 0; p < np; ++p) {
            std::snprintf(names.data() + 64 * p, 64, "%s", gm.patch_names[p].c_str());
            std::snprintf(types.data() + 64 * p, 64, "%s", gm.patch_types[p].c_str());
        }
        h5_write(file, "patch_names", H5T_NATIVE_CHAR, names.data(), {(hsize_t)np, 64});
        h5_write(file, "patch_types", H5T_NATIVE_CHAR, types.data(), {(hsize_t)np, 64});

        hid_t g = h5_make_group(file, "cells");
        h5_write_zero(g, "type", H5T_NATIVE_UINT8, {(hsize_t)cells_g});
        h5_write_zero(g, "gid", H5T_NATIVE_INT32, {(hsize_t)cells_g});
        h5_write_zero(g, "donor", H5T_NATIVE_INT32, {(hsize_t)cells_g});
        h5_write_zero(g, "nodes", H5T_NATIVE_INT32, {(hsize_t)cells_g, 8});
        h5_write_zero(g, "centroid", H5T_NATIVE_DOUBLE, {(hsize_t)cells_g, 3});
        h5_write_zero(g, "volume", H5T_NATIVE_DOUBLE, {(hsize_t)cells_g});
        H5Gclose(g);

        g = h5_make_group(file, "nodes");
        h5_write_zero(g, "xyz", H5T_NATIVE_DOUBLE, {(hsize_t)nodes_g, 3});
        h5_write_zero(g, "gid", H5T_NATIVE_INT32, {(hsize_t)nodes_g});
        H5Gclose(g);

        g = h5_make_group(file, "faces");
        h5_write_zero(g, "owner", H5T_NATIVE_INT32, {(hsize_t)faces_g});
        h5_write_zero(g, "neigh", H5T_NATIVE_INT32, {(hsize_t)faces_g});
        h5_write_zero(g, "type", H5T_NATIVE_UINT8, {(hsize_t)faces_g});
        h5_write_zero(g, "nodes", H5T_NATIVE_INT32, {(hsize_t)faces_g, 4});
        h5_write_zero(g, "centroid", H5T_NATIVE_DOUBLE, {(hsize_t)faces_g, 3});
        h5_write_zero(g, "normal", H5T_NATIVE_DOUBLE, {(hsize_t)faces_g, 3});
        h5_write_zero(g, "area", H5T_NATIVE_DOUBLE, {(hsize_t)faces_g});
        h5_write_zero(g, "patch", H5T_NATIVE_INT32, {(hsize_t)faces_g});
        h5_write_zero(g, "donor", H5T_NATIVE_INT32, {(hsize_t)faces_g});
        H5Gclose(g);

        g = h5_make_group(file, "comm");
        h5_write_zero(g, "nb_ranks", H5T_NATIVE_INT32, {(hsize_t)nb_g});
        h5_write_zero(g, "recv_ghost_local", H5T_NATIVE_INT32, {(hsize_t)recv_g});
        h5_write_zero(g, "send_owned_local", H5T_NATIVE_INT32, {(hsize_t)send_g});
        h5_write_zero(g, "recv_seg", H5T_NATIVE_INT32, {(hsize_t)rseg_g});
        h5_write_zero(g, "send_seg", H5T_NATIVE_INT32, {(hsize_t)sseg_g});
        H5Gclose(g);

        g = h5_make_group(file, "patches");
        h5_write_zero(g, "face_idx", H5T_NATIVE_INT32, {(hsize_t)pf_g});
        H5Gclose(g);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Phase 2: all ranks write their hyperslabs simultaneously (collective) ---
    H5Obj file = h5_open_file_rdwr(path);

    hid_t g = h5_open_group(file, "cells");
    h5_write_slab(g, "type", H5T_NATIVE_UINT8, mp.cell_type.data(), cells_b, {(hsize_t)ncells_l});
    h5_write_slab(g, "gid", H5T_NATIVE_INT32, mp.cell_gid.data(), cells_b, {(hsize_t)ncells_l});
    h5_write_slab(g, "donor", H5T_NATIVE_INT32, mp.cell_donor.data(), cells_b, {(hsize_t)ncells_l});
    h5_write_slab(g, "nodes", H5T_NATIVE_INT32, mp.cell_nodes.data(), cells_b,
                  {(hsize_t)ncells_l, 8});
    h5_write_slab(g, "centroid", H5T_NATIVE_DOUBLE, mp.cell_centroid.data(), cells_b,
                  {(hsize_t)ncells_l, 3});
    h5_write_slab(g, "volume", H5T_NATIVE_DOUBLE, mp.cell_volume.data(), cells_b,
                  {(hsize_t)ncells_l});
    H5Gclose(g);

    g = h5_open_group(file, "nodes");
    h5_write_slab(g, "xyz", H5T_NATIVE_DOUBLE, mp.node_xyz.data(), nodes_b, {(hsize_t)nnodes_l, 3});
    h5_write_slab(g, "gid", H5T_NATIVE_INT32, mp.node_gid.data(), nodes_b, {(hsize_t)nnodes_l});
    H5Gclose(g);

    g = h5_open_group(file, "faces");
    h5_write_slab(g, "owner", H5T_NATIVE_INT32, mp.face_owner.data(), faces_b, {(hsize_t)nfaces_l});
    h5_write_slab(g, "neigh", H5T_NATIVE_INT32, mp.face_neigh.data(), faces_b, {(hsize_t)nfaces_l});
    h5_write_slab(g, "type", H5T_NATIVE_UINT8, mp.face_type.data(), faces_b, {(hsize_t)nfaces_l});
    h5_write_slab(g, "nodes", H5T_NATIVE_INT32, mp.face_nodes.data(), faces_b,
                  {(hsize_t)nfaces_l, 4});
    h5_write_slab(g, "centroid", H5T_NATIVE_DOUBLE, mp.face_centroid.data(), faces_b,
                  {(hsize_t)nfaces_l, 3});
    h5_write_slab(g, "normal", H5T_NATIVE_DOUBLE, mp.face_normal.data(), faces_b,
                  {(hsize_t)nfaces_l, 3});
    h5_write_slab(g, "area", H5T_NATIVE_DOUBLE, mp.face_area.data(), faces_b, {(hsize_t)nfaces_l});
    h5_write_slab(g, "patch", H5T_NATIVE_INT32, mp.face_patch.data(), faces_b, {(hsize_t)nfaces_l});
    h5_write_slab(g, "donor", H5T_NATIVE_INT32, mp.face_donor.data(), faces_b, {(hsize_t)nfaces_l});
    H5Gclose(g);

    g = h5_open_group(file, "comm");
    h5_write_slab(g, "nb_ranks", H5T_NATIVE_INT32, mp.nb_ranks.data(), nb_b, {(hsize_t)nnb_l});
    h5_write_slab(g, "recv_ghost_local", H5T_NATIVE_INT32, mp.recv_ghost_local.data(), recv_b,
                  {(hsize_t)nrecv_l});
    h5_write_slab(g, "send_owned_local", H5T_NATIVE_INT32, mp.send_owned_local.data(), send_b,
                  {(hsize_t)nsend_l});
    h5_write_slab(g, "recv_seg", H5T_NATIVE_INT32, mp.recv_offsets.data(), rseg_b,
                  {(hsize_t)nseg_l});
    h5_write_slab(g, "send_seg", H5T_NATIVE_INT32, mp.send_offsets.data(), sseg_b,
                  {(hsize_t)nseg_l});
    H5Gclose(g);

    g = h5_open_group(file, "patches");
    h5_write_slab(g, "face_idx", H5T_NATIVE_INT32, mp.patch_faces.data(), pf_b, {(hsize_t)npf_l});
    H5Gclose(g);

    // Close the file collectively.
    const hid_t fid = file.release();
    MPI_Barrier(MPI_COMM_WORLD);
    if (H5Fclose(fid) < 0) log_warn_rank("H5Fclose failed — the file may be incomplete");
    log_stat("HDF5 written: %s", path.c_str());
}

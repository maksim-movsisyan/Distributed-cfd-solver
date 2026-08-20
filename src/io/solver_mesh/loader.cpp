#include "cfd/io/solver_mesh/loader.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd/io/solver_mesh/h5util.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

GlobalMeta load_global_meta(hid_t file) {
    GlobalMeta gm;
    gm.nprocs = h5_attr_i(file, "nprocs");
    gm.n_cells_g = h5_attr_l(file, "n_cells_g");
    gm.n_faces_g = h5_attr_l(file, "n_faces_g");
    gm.n_nodes_g = h5_attr_l(file, "n_nodes_g");
    gm.n_bfaces_g = h5_attr_l(file, "n_bfaces_g");
    gm.total_volume = h5_attr_d(file, "total_volume");
    gm.n_flipped = h5_attr_l(file, "n_flipped");
    gm.n_ghost_total = h5_attr_l(file, "n_ghost_total");
    {
        H5Obj at(H5Aopen(file, "bbox_lo", H5P_DEFAULT), H5Obj::Kind::Attr);
        if (at.valid()) H5Aread(at, H5T_NATIVE_DOUBLE, gm.bbox_lo);
        H5Obj at2(H5Aopen(file, "bbox_hi", H5P_DEFAULT), H5Obj::Kind::Attr);
        if (at2.valid()) H5Aread(at2, H5T_NATIVE_DOUBLE, gm.bbox_hi);
    }
    const size_t np1 = static_cast<size_t>(gm.nprocs) + 1;
    gm.cells_off = h5_read_l(file, "cells_off", np1);
    gm.nodes_off = h5_read_l(file, "nodes_off", np1);
    gm.faces_off = h5_read_l(file, "faces_off", np1);
    gm.nb_off = h5_read_l(file, "nb_off", np1);
    gm.recv_off = h5_read_l(file, "recv_off", np1);
    gm.send_off = h5_read_l(file, "send_off", np1);
    gm.patchfaces_off = h5_read_l(file, "patchfaces_off", np1);
    gm.recvseg_off = h5_read_l(file, "recvseg_off", np1);
    gm.sendseg_off = h5_read_l(file, "sendseg_off", np1);
    gm.n_own = h5_read_l(file, "n_own", static_cast<size_t>(gm.nprocs));

    hsize_t dims[2] = {0, 0};
    H5Obj ds(H5Dopen(file, "patch_names", H5P_DEFAULT), H5Obj::Kind::Dataset);
    if (ds.valid()) {
        H5Obj sp(H5Dget_space(ds), H5Obj::Kind::Space);
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        std::vector<char> buf(dims[0] * dims[1]);
        H5Dread(ds, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        for (hsize_t p = 0; p < dims[0]; ++p) gm.patch_names.emplace_back(buf.data() + p * dims[1]);
        H5Obj ds2(H5Dopen(file, "patch_types", H5P_DEFAULT), H5Obj::Kind::Dataset);
        H5Dread(ds2, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        for (hsize_t p = 0; p < dims[0]; ++p) gm.patch_types.emplace_back(buf.data() + p * dims[1]);
    }
    return gm;
}

void load_partition(hid_t file, int rank, const GlobalMeta& gm, MeshPart& mp) {
    mp.rank = rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mp.nprocs);

    // Ranges of my part inside the flat arrays.
    const hsize_t cb = (hsize_t)gm.cells_off[rank];
    const hsize_t nb_ = (hsize_t)gm.nodes_off[rank];
    const hsize_t fb = (hsize_t)gm.faces_off[rank];
    const hsize_t nbb = (hsize_t)gm.nb_off[rank];
    const hsize_t rb = (hsize_t)gm.recv_off[rank];
    const hsize_t sb = (hsize_t)gm.send_off[rank];
    const hsize_t pb = (hsize_t)gm.patchfaces_off[rank];

    mp.n_own = static_cast<int>(gm.n_own[rank]);
    mp.n_cells = static_cast<int>(gm.cells_off[rank + 1] - gm.cells_off[rank]);
    mp.n_nodes = static_cast<int>(gm.nodes_off[rank + 1] - gm.nodes_off[rank]);
    mp.n_faces = static_cast<int>(gm.faces_off[rank + 1] - gm.faces_off[rank]);
    const int nnb = static_cast<int>(gm.nb_off[rank + 1] - gm.nb_off[rank]);
    const int nrecv = static_cast<int>(gm.recv_off[rank + 1] - gm.recv_off[rank]);
    const int nsend = static_cast<int>(gm.send_off[rank + 1] - gm.send_off[rank]);
    const int npf = static_cast<int>(gm.patchfaces_off[rank + 1] - gm.patchfaces_off[rank]);

    // Patches: the global list.
    mp.patches.clear();
    for (size_t p = 0; p < gm.patch_names.size(); ++p)
        mp.patches.push_back(
            {gm.patch_names[p], p < gm.patch_types.size() ? gm.patch_types[p] : ""});

    hid_t g = h5_open_group(file, "cells");
    mp.cell_type.resize(mp.n_cells);
    h5_read_slab(g, "type", H5T_NATIVE_UINT8, mp.cell_type.data(), cb, {(hsize_t)mp.n_cells});
    mp.cell_gid.resize(mp.n_cells);
    h5_read_slab(g, "gid", H5T_NATIVE_INT32, mp.cell_gid.data(), cb, {(hsize_t)mp.n_cells});
    mp.cell_donor.resize(mp.n_cells);
    h5_read_slab(g, "donor", H5T_NATIVE_INT32, mp.cell_donor.data(), cb, {(hsize_t)mp.n_cells});
    mp.cell_nodes.resize((size_t)mp.n_cells * 8);
    h5_read_slab(g, "nodes", H5T_NATIVE_INT32, mp.cell_nodes.data(), cb, {(hsize_t)mp.n_cells, 8});
    mp.cell_centroid.resize((size_t)mp.n_cells * 3);
    h5_read_slab(g, "centroid", H5T_NATIVE_DOUBLE, mp.cell_centroid.data(), cb,
                 {(hsize_t)mp.n_cells, 3});
    mp.cell_volume.resize(mp.n_cells);
    h5_read_slab(g, "volume", H5T_NATIVE_DOUBLE, mp.cell_volume.data(), cb, {(hsize_t)mp.n_cells});
    H5Gclose(g);

    g = h5_open_group(file, "nodes");
    mp.node_xyz.resize((size_t)mp.n_nodes * 3);
    h5_read_slab(g, "xyz", H5T_NATIVE_DOUBLE, mp.node_xyz.data(), nb_, {(hsize_t)mp.n_nodes, 3});
    mp.node_gid.resize(mp.n_nodes);
    h5_read_slab(g, "gid", H5T_NATIVE_INT32, mp.node_gid.data(), nb_, {(hsize_t)mp.n_nodes});
    H5Gclose(g);
    mp.n_nodes_own = 0;  // not stored explicitly; n_nodes is enough for the solver

    g = h5_open_group(file, "faces");
    mp.face_owner.resize(mp.n_faces);
    h5_read_slab(g, "owner", H5T_NATIVE_INT32, mp.face_owner.data(), fb, {(hsize_t)mp.n_faces});
    mp.face_neigh.resize(mp.n_faces);
    h5_read_slab(g, "neigh", H5T_NATIVE_INT32, mp.face_neigh.data(), fb, {(hsize_t)mp.n_faces});
    mp.face_type.resize(mp.n_faces);
    h5_read_slab(g, "type", H5T_NATIVE_UINT8, mp.face_type.data(), fb, {(hsize_t)mp.n_faces});
    mp.face_nodes.resize((size_t)mp.n_faces * 4);
    h5_read_slab(g, "nodes", H5T_NATIVE_INT32, mp.face_nodes.data(), fb, {(hsize_t)mp.n_faces, 4});
    mp.face_centroid.resize((size_t)mp.n_faces * 3);
    h5_read_slab(g, "centroid", H5T_NATIVE_DOUBLE, mp.face_centroid.data(), fb,
                 {(hsize_t)mp.n_faces, 3});
    mp.face_normal.resize((size_t)mp.n_faces * 3);
    h5_read_slab(g, "normal", H5T_NATIVE_DOUBLE, mp.face_normal.data(), fb,
                 {(hsize_t)mp.n_faces, 3});
    mp.face_area.resize(mp.n_faces);
    h5_read_slab(g, "area", H5T_NATIVE_DOUBLE, mp.face_area.data(), fb, {(hsize_t)mp.n_faces});
    mp.face_patch.resize(mp.n_faces);
    h5_read_slab(g, "patch", H5T_NATIVE_INT32, mp.face_patch.data(), fb, {(hsize_t)mp.n_faces});
    mp.face_donor.resize(mp.n_faces);
    h5_read_slab(g, "donor", H5T_NATIVE_INT32, mp.face_donor.data(), fb, {(hsize_t)mp.n_faces});
    H5Gclose(g);

    g = h5_open_group(file, "comm");
    mp.nb_ranks.resize(nnb);
    h5_read_slab(g, "nb_ranks", H5T_NATIVE_INT32, mp.nb_ranks.data(), nbb, {(hsize_t)nnb});
    mp.recv_ghost_local.resize(nrecv);
    h5_read_slab(g, "recv_ghost_local", H5T_NATIVE_INT32, mp.recv_ghost_local.data(), rb,
                 {(hsize_t)nrecv});
    mp.send_owned_local.resize(nsend);
    h5_read_slab(g, "send_owned_local", H5T_NATIVE_INT32, mp.send_owned_local.data(), sb,
                 {(hsize_t)nsend});
    mp.recv_offsets.resize(nnb + 1);
    h5_read_slab(g, "recv_seg", H5T_NATIVE_INT32, mp.recv_offsets.data(),
                 (hsize_t)gm.recvseg_off[rank], {(hsize_t)(nnb + 1)});
    mp.send_offsets.resize(nnb + 1);
    h5_read_slab(g, "send_seg", H5T_NATIVE_INT32, mp.send_offsets.data(),
                 (hsize_t)gm.sendseg_off[rank], {(hsize_t)(nnb + 1)});
    H5Gclose(g);

    g = h5_open_group(file, "patches");
    mp.patch_faces.resize(npf);
    h5_read_slab(g, "face_idx", H5T_NATIVE_INT32, mp.patch_faces.data(), pb, {(hsize_t)npf});
    H5Gclose(g);

    // patch offsets: my face count per patch.
    const size_t np = gm.patch_names.size();
    mp.patch_face_offsets.assign(np + 1, 0);
    {
        std::vector<int> cnt(np, 0);
        for (int i = 0; i < mp.n_faces; ++i)
            if (mp.face_patch[i] >= 0 && (size_t)mp.face_patch[i] < np) ++cnt[mp.face_patch[i]];
        for (size_t p = 0; p < np; ++p)
            mp.patch_face_offsets[p + 1] = mp.patch_face_offsets[p] + cnt[p];
    }
}

bool verify_partition(const MeshPart& mp, const GlobalMeta& gm) {
    bool ok = true;
    std::string err;
    if (!meshpart_sane(mp, err)) {
        log_rank("invariants violated: %s", err.c_str());
        ok = false;
    }

    // --- Ghost-map handshake ---
    // For each pair (me r, neighbour q): my recv from q must equal q's send to r
    // (the gid sequences must match element-wise), and vice versa.
    for (int n = 0; n < mp.n_neighbors(); ++n) {
        const int q = mp.nb_ranks[n];
        // The tag must be symmetric for the pair (rank, q): with 700+peer
        // the two sides wait with different tags — a deadlock.
        const int lo = std::min(mp.rank, q), hi = std::max(mp.rank, q);
        const int tag_cnt = 700000 + lo * 1000 + hi;
        const int tag_gid = tag_cnt + 500000;
        const int my_recv = mp.recv_offsets[n + 1] - mp.recv_offsets[n];
        const int my_send = mp.send_offsets[n + 1] - mp.send_offsets[n];
        int his[2] = {-1, -1};
        const int mine[2] = {my_recv, my_send};
        MPI_Sendrecv(mine, 2, MPI_INT, q, tag_cnt, his, 2, MPI_INT, q, tag_cnt, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        // mine[0] = my ghosts from q; his[1] = his send to me; mine[1] = my send
        // to him; his[0] = his ghosts from me.
        if (my_recv != his[1] || my_send != his[0]) {
            log_rank(
                "ghost maps with %d: my recv %d / his send %d; my send %d / "
                "his recv %d",
                q, my_recv, his[1], my_send, his[0]);
            ok = false;
            continue;
        }
        // Exchange OWN send lists (gids) and compare the neighbour's send
        // against my recv: the sequences must match element-wise.
        std::vector<int32_t> my_send_gid(my_send), his_send_gid(my_recv);
        for (int i = 0; i < my_send; ++i)
            my_send_gid[i] = mp.cell_gid[mp.send_owned_local[mp.send_offsets[n] + i]];
        MPI_Sendrecv(my_send_gid.data(), my_send, MPI_INT, q, tag_gid, his_send_gid.data(), my_recv,
                     MPI_INT, q, tag_gid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < my_recv; ++i) {
            const int32_t my_gid = mp.cell_gid[mp.recv_ghost_local[mp.recv_offsets[n] + i]];
            if (my_gid != his_send_gid[i]) {
                log_rank(
                    "ghost maps with %d, position %d: my recv gid %d != his send "
                    "gid %d",
                    q, i, my_gid, his_send_gid[i]);
                ok = false;
                break;
            }
        }
    }

    // --- Global sum of owned-cell volumes ---
    double vsum = 0.0;
    for (int i = 0; i < mp.n_own; ++i) vsum += mp.cell_volume[i];
    const double vtot = d_sum(vsum);
    const double relerr =
        gm.total_volume != 0.0 ? std::fabs(vtot - gm.total_volume) / gm.total_volume : 0.0;
    if (relerr > 1e-12) {
        log_rank("volume sum %.17g != stored %.17g (rel %.2e)", vtot, gm.total_volume, relerr);
        ok = false;
    }

    // --- Totals (ll_sum values are already global) ---
    const long long nc = ll_sum(mp.n_own);
    const long long nf = ll_sum(mp.n_faces);
    const long long ng = ll_sum(mp.n_cells - mp.n_own);
    if (mp.rank == 0 && (nc != gm.n_cells_g || nf != gm.n_faces_g || ng != gm.n_ghost_total)) {
        log_rank(
            "totals: cells %lld/%lld, faces %lld/%lld, ghosts "
            "%lld/%lld",
            nc, gm.n_cells_g, nf, gm.n_faces_g, ng, gm.n_ghost_total);
    }
    ok = ok && nc == gm.n_cells_g && nf == gm.n_faces_g && ng == gm.n_ghost_total;
    return ok;
}

void print_mesh_stats(const MeshPart& mp, const GlobalMeta& gm) {
    if (mp.rank == 0) {
        log_info("=== Loaded mesh metadata ===");
        log_info("ranks: %d", gm.nprocs);
        log_info("cells (global): %lld", gm.n_cells_g);
        log_info("faces (global): %lld (boundary %lld)", gm.n_faces_g, gm.n_bfaces_g);
        log_info("nodes (global): %lld", gm.n_nodes_g);
        log_info("ghost cells (total): %lld (%.2f%% of owned)", gm.n_ghost_total,
                 gm.n_cells_g ? 100.0 * gm.n_ghost_total / gm.n_cells_g : 0.0);
        log_info("volume sum: %.10g", gm.total_volume);
        log_info("fixed orientations: %lld", gm.n_flipped);
        log_info("bbox: [%.3g %.3g %.3g] .. [%.3g %.3g %.3g]", gm.bbox_lo[0], gm.bbox_lo[1],
                 gm.bbox_lo[2], gm.bbox_hi[0], gm.bbox_hi[1], gm.bbox_hi[2]);
        log_info("BC patches: %zu", gm.patch_names.size());
        for (size_t p = 0; p < gm.patch_names.size(); ++p)
            log_info("  patch %zu: '%s' (cgns: %s)", p, gm.patch_names[p].c_str(),
                     gm.patch_types[p].c_str());
    }
    if (!log_verbose()) return;
    long long nb_bnd = 0, nb_iface = 0;
    for (int i = 0; i < mp.n_faces; ++i) {
        nb_bnd += (mp.face_neigh[i] < 0);
        nb_iface += (mp.face_donor[i] >= 0);
    }
    std::vector<long long> per_patch(mp.patches.size(), 0);
    for (int i = 0; i < mp.n_faces; ++i)
        if (mp.face_patch[i] >= 0) ++per_patch[mp.face_patch[i]];
    log_rank(
        "owned=%d ghosts=%d nodes=%d faces=%d (boundary %lld, "
        "interface %lld) neighbours=%d",
        mp.n_own, mp.n_cells - mp.n_own, mp.n_nodes, mp.n_faces, nb_bnd, nb_iface,
        mp.n_neighbors());
    for (size_t p = 0; p < per_patch.size(); ++p)
        log_rank("  patch %zu ('%s'): %lld faces", p, mp.patches[p].name.c_str(), per_patch[p]);
}

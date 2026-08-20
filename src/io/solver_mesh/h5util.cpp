#include "cfd/io/solver_mesh/h5util.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <cstdio>

#include "cfd/mpi/log.hpp"

void H5Obj::close() {
    if (id_ >= 0) {
        switch (kind_) {
            case Kind::Dataset:
                H5Dclose(id_);
                break;
            case Kind::Group:
                H5Gclose(id_);
                break;
            case Kind::Attr:
                H5Aclose(id_);
                break;
            case Kind::Space:
                H5Sclose(id_);
                break;
            case Kind::Plist:
                H5Pclose(id_);
                break;
            case Kind::File:
                H5Fclose(id_);
                break;
        }
        id_ = -1;
    }
}

hid_t h5_type_of(double) { return H5T_NATIVE_DOUBLE; }
hid_t h5_type_of(float) { return H5T_NATIVE_FLOAT; }
hid_t h5_type_of(int32_t) { return H5T_NATIVE_INT32; }
hid_t h5_type_of(int64_t) { return H5T_NATIVE_INT64; }
hid_t h5_type_of(uint8_t) { return H5T_NATIVE_UINT8; }

H5Obj h5_create_file(const std::string& path) {
    H5Obj fa(H5Pcreate(H5P_FILE_ACCESS), H5Obj::Kind::Plist);
    H5Pset_fapl_mpio(fa, MPI_COMM_WORLD, MPI_INFO_NULL);
    H5Obj fc(H5Pcreate(H5P_FILE_CREATE), H5Obj::Kind::Plist);
    H5Obj f(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, fc, fa), H5Obj::Kind::File);
    if (!f.valid()) {
        log_rank("H5Fcreate failed: %s", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return f;
}

H5Obj h5_open_file_rdonly(const std::string& path) {
    H5Obj fa(H5Pcreate(H5P_FILE_ACCESS), H5Obj::Kind::Plist);
    H5Pset_fapl_mpio(fa, MPI_COMM_WORLD, MPI_INFO_NULL);
    // No all_coll_metadata_ops: unsynchronized per-rank reads stay safe
    // (the file never changes), while the collective mode hit errors at close.
    // close.
    H5Obj f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, fa), H5Obj::Kind::File);
    if (!f.valid()) {
        log_rank("H5Fopen failed: %s", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return f;
}

H5Obj h5_open_file_rdwr(const std::string& path) {
    H5Obj fa(H5Pcreate(H5P_FILE_ACCESS), H5Obj::Kind::Plist);
    H5Pset_fapl_mpio(fa, MPI_COMM_WORLD, MPI_INFO_NULL);
    // Collective metadata: all ranks go through the file close along the same
    // path (no desynchronized collectives inside H5Fclose).
    H5Pset_coll_metadata_write(fa, true);
    H5Pset_all_coll_metadata_ops(fa, true);
    H5Obj f(H5Fopen(path.c_str(), H5F_ACC_RDWR, fa), H5Obj::Kind::File);
    if (!f.valid()) {
        log_rank("H5Fopen(RDWR) failed: %s", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return f;
}

H5Obj h5_create_file_serial(const std::string& path) {
    // Serial driver (sec2): call on ONE rank only while nobody else has the
    // file open; used to create the metadata skeleton.
    H5Obj f(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Obj::Kind::File);
    if (!f.valid()) {
        log_rank("H5Fcreate(serial) failed: %s", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return f;
}

hid_t h5_make_group(hid_t loc, const std::string& name) {
    hid_t g = H5Gcreate(loc, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0) {
        log_rank("H5Gcreate failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return g;
}

hid_t h5_open_group(hid_t loc, const std::string& name) {
    hid_t g = H5Gopen(loc, name.c_str(), H5P_DEFAULT);
    if (g < 0) {
        log_rank("H5Gopen failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return g;
}

void h5_write(hid_t loc, const std::string& name, hid_t htype, const void* data,
              const std::vector<hsize_t>& dims) {
    hsize_t total = 1;
    for (hsize_t d : dims) total *= d;
    h5_create_ds(loc, name, htype, dims);
    if (total > 0) {
        H5Obj ds(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Dataset);
        if (!ds.valid()) {
            log_rank("H5Dopen failed: %s", name.c_str());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        if (H5Dwrite(ds, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
            log_rank("H5Dwrite failed: %s", name.c_str());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }
}

void h5_create_ds(hid_t loc, const std::string& name, hid_t htype,
                  const std::vector<hsize_t>& dims) {
    hsize_t total = 1;
    for (hsize_t d : dims) total *= d;
    H5Obj sp;
    if (total == 0)
        sp = H5Obj(H5Screate(H5S_NULL), H5Obj::Kind::Space);  // empty dataset
    else
        sp = H5Obj(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
                   H5Obj::Kind::Space);
    H5Obj ds(H5Dcreate(loc, name.c_str(), htype, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
             H5Obj::Kind::Dataset);
    if (!ds.valid()) {
        log_rank("H5Dcreate failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

void h5_write_zero(hid_t loc, const std::string& name, hid_t htype,
                   const std::vector<hsize_t>& dims) {
    hsize_t total = 1;
    for (hsize_t d : dims) total *= d;
    h5_create_ds(loc, name, htype, dims);
    if (total == 0) return;
    const hsize_t bytes = total * H5Tget_size(htype);
    std::vector<char> zeros(bytes, 0);
    h5_write_existing(loc, name, htype, zeros.data(), dims);
}

void h5_write_existing(hid_t loc, const std::string& name, hid_t htype, const void* data,
                       const std::vector<hsize_t>& dims) {
    hsize_t total = 1;
    for (hsize_t d : dims) total *= d;
    if (total == 0) return;  // empty datasets stay empty
    H5Obj ds(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Dataset);
    if (!ds.valid()) {
        log_rank("H5Dopen(write) failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (H5Dwrite(ds, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        log_rank("H5Dwrite failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

void h5_write_slab(hid_t loc, const std::string& name, hid_t htype, const void* data, hsize_t start,
                   const std::vector<hsize_t>& local_dims) {
    hsize_t total = 1;
    for (hsize_t d : local_dims) total *= d;
    H5Obj ds(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Dataset);
    if (!ds.valid()) {
        log_rank("H5Dopen(slab) failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (total == 0) {
        // Empty contribution: still participate in the collective write.
        H5Obj dx(H5Pcreate(H5P_DATASET_XFER), H5Obj::Kind::Plist);
        H5Pset_dxpl_mpio(dx, H5FD_MPIO_COLLECTIVE);
        H5Obj mem(H5Screate(H5S_NULL), H5Obj::Kind::Space);
        H5Dwrite(ds, htype, mem, H5S_ALL, dx, nullptr);
        return;
    }
    H5Obj fs(H5Dget_space(ds), H5Obj::Kind::Space);
    hsize_t fdims[H5S_MAX_RANK];
    const int rank = H5Sget_simple_extent_dims(fs, fdims, nullptr);
    hsize_t fstart[H5S_MAX_RANK], fcount[H5S_MAX_RANK];
    for (int d = 0; d < rank; ++d) {
        fstart[d] = 0;
        fcount[d] = fdims[d];
    }
    if (rank > 0) {
        fstart[0] = start;
        fcount[0] = local_dims[0];
    }
    H5Sselect_hyperslab(fs, H5S_SELECT_SET, fstart, nullptr, fcount, nullptr);
    H5Obj ms(H5Screate_simple(static_cast<int>(local_dims.size()), local_dims.data(), nullptr),
             H5Obj::Kind::Space);
    H5Obj dx(H5Pcreate(H5P_DATASET_XFER), H5Obj::Kind::Plist);
    H5Pset_dxpl_mpio(dx, H5FD_MPIO_COLLECTIVE);
    if (H5Dwrite(ds, htype, ms, fs, dx, data) < 0) {
        log_rank(
            "H5Dwrite(slab) failed: %s (start=%llu, local=[%llu x %llu], "
            "file rank=%d fdims=[%llu x %llu], mrank=%d)",
            name.c_str(), (unsigned long long)start,
            (unsigned long long)(local_dims.size() > 0 ? local_dims[0] : 0),
            (unsigned long long)(local_dims.size() > 1 ? local_dims[1] : 0), rank,
            (unsigned long long)(rank > 0 ? fdims[0] : 0),
            (unsigned long long)(rank > 1 ? fdims[1] : 0), (int)local_dims.size());
        H5Eprint2(H5E_DEFAULT, stderr);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

void h5_read_slab(hid_t loc, const std::string& name, hid_t htype, void* data, hsize_t start,
                  const std::vector<hsize_t>& local_dims) {
    hsize_t total = 1;
    for (hsize_t d : local_dims) total *= d;
    if (total == 0) return;
    H5Obj ds(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Dataset);
    if (!ds.valid()) {
        log_rank("H5Dopen(slab rd) failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    H5Obj fs(H5Dget_space(ds), H5Obj::Kind::Space);
    hsize_t fdims[H5S_MAX_RANK];
    const int rank = H5Sget_simple_extent_dims(fs, fdims, nullptr);
    hsize_t fstart[H5S_MAX_RANK], fcount[H5S_MAX_RANK];
    for (int d = 0; d < rank; ++d) {
        fstart[d] = 0;
        fcount[d] = fdims[d];
    }
    if (rank > 0) {
        fstart[0] = start;
        fcount[0] = local_dims[0];
    }
    H5Sselect_hyperslab(fs, H5S_SELECT_SET, fstart, nullptr, fcount, nullptr);
    H5Obj ms(H5Screate_simple(static_cast<int>(local_dims.size()), local_dims.data(), nullptr),
             H5Obj::Kind::Space);
    if (H5Dread(ds, htype, ms, fs, H5P_DEFAULT, data) < 0) {
        log_rank("H5Dread(slab) failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

void h5_read(hid_t loc, const std::string& name, hid_t htype, void* data,
             const std::vector<hsize_t>& dims) {
    hsize_t total = 1;
    for (hsize_t d : dims) total *= d;
    H5Obj ds(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Dataset);
    if (!ds.valid()) {
        log_rank("H5Dopen failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (total > 0) {
        if (H5Dread(ds, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
            log_rank("H5Dread failed: %s", name.c_str());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }
}

std::vector<double> h5_read_d(hid_t loc, const std::string& name, size_t expect) {
    std::vector<double> v(expect);
    h5_read(loc, name, H5T_NATIVE_DOUBLE, v.data(), {static_cast<hsize_t>(expect)});
    return v;
}
std::vector<int32_t> h5_read_i(hid_t loc, const std::string& name, size_t expect) {
    std::vector<int32_t> v(expect);
    h5_read(loc, name, H5T_NATIVE_INT32, v.data(), {static_cast<hsize_t>(expect)});
    return v;
}
std::vector<int64_t> h5_read_l(hid_t loc, const std::string& name, size_t expect) {
    std::vector<int64_t> v(expect);
    h5_read(loc, name, H5T_NATIVE_INT64, v.data(), {static_cast<hsize_t>(expect)});
    return v;
}
std::vector<uint8_t> h5_read_u8(hid_t loc, const std::string& name, size_t expect) {
    std::vector<uint8_t> v(expect);
    h5_read(loc, name, H5T_NATIVE_UINT8, v.data(), {static_cast<hsize_t>(expect)});
    return v;
}

void h5_attr(hid_t loc, const std::string& name, hid_t htype, const void* v) {
    H5Obj sp(H5Screate(H5S_SCALAR), H5Obj::Kind::Space);
    H5Obj at(H5Acreate(loc, name.c_str(), htype, sp, H5P_DEFAULT, H5P_DEFAULT), H5Obj::Kind::Attr);
    if (H5Awrite(at, htype, v) < 0) {
        log_rank("H5Awrite failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

void h5_attr_vec(hid_t loc, const std::string& name, hid_t htype, const void* v, size_t n) {
    hsize_t dim = n;
    H5Obj sp(H5Screate_simple(1, &dim, nullptr), H5Obj::Kind::Space);
    H5Obj at(H5Acreate(loc, name.c_str(), htype, sp, H5P_DEFAULT, H5P_DEFAULT), H5Obj::Kind::Attr);
    if (H5Awrite(at, htype, v) < 0) {
        log_rank("H5Awrite failed: %s", name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
}

double h5_attr_d(hid_t loc, const std::string& name) {
    double v = 0;
    H5Obj at(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Attr);
    if (at.valid()) H5Aread(at, H5T_NATIVE_DOUBLE, &v);
    return v;
}
long long h5_attr_l(hid_t loc, const std::string& name) {
    long long v = 0;
    H5Obj at(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Attr);
    if (at.valid()) H5Aread(at, H5T_NATIVE_INT64, &v);
    return v;
}
int h5_attr_i(hid_t loc, const std::string& name) {
    int v = 0;
    H5Obj at(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Obj::Kind::Attr);
    if (at.valid()) H5Aread(at, H5T_NATIVE_INT32, &v);
    return v;
}

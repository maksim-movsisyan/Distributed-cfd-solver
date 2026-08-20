#pragma once
// Thin wrappers over the HDF5 C API (RAII plus simple dataset IO).

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

// RAII wrapper: closes the hid_t with the right function per object kind.
class H5Obj {
public:
    enum class Kind { Dataset, Group, Attr, Space, Plist, File };
    H5Obj() = default;
    H5Obj(hid_t id, Kind kind) : id_(id), kind_(kind) {}
    H5Obj(const H5Obj&) = delete;
    H5Obj& operator=(const H5Obj&) = delete;
    H5Obj(H5Obj&& o) noexcept { swap(o); }
    H5Obj& operator=(H5Obj&& o) noexcept {
        if (this != &o) {
            close();
            swap(o);
        }
        return *this;
    }
    ~H5Obj() { close(); }
    hid_t get() const { return id_; }
    operator hid_t() const { return id_; }
    bool valid() const { return id_ >= 0; }
    // Give up ownership (close manually).
    hid_t release() {
        hid_t v = id_;
        id_ = -1;
        return v;
    }

private:
    void swap(H5Obj& o) {
        std::swap(id_, o.id_);
        std::swap(kind_, o.kind_);
    }
    void close();
    hid_t id_ = -1;
    Kind kind_ = Kind::Dataset;
};

hid_t h5_type_of(double);
hid_t h5_type_of(float);
hid_t h5_type_of(int32_t);
hid_t h5_type_of(int64_t);
hid_t h5_type_of(uint8_t);

// Create/open a file in parallel (on MPI_COMM_WORLD).
H5Obj h5_create_file(const std::string& path);
H5Obj h5_open_file_rdonly(const std::string& path);
// Open an existing file in parallel for writing (RDWR).
H5Obj h5_open_file_rdwr(const std::string& path);
// Create the file on rank 0 ONLY (serial driver, no MPI-IO) for the metadata
// skeleton; other ranks must not touch the file meanwhile.
H5Obj h5_create_file_serial(const std::string& path);

// Group (create/open): raw hid_t, closed manually via H5Gclose.
hid_t h5_make_group(hid_t loc, const std::string& name);
hid_t h5_open_group(hid_t loc, const std::string& name);

// Write dataset name into loc (dims define the shape).
void h5_write(hid_t loc, const std::string& name, hid_t htype, const void* data,
              const std::vector<hsize_t>& dims);

// Create an empty dataset of the given shape (skeleton; phase 2 writes data).
void h5_create_ds(hid_t loc, const std::string& name, hid_t htype,
                  const std::vector<hsize_t>& dims);

// Create a dataset and WRITE zeros (full storage allocation in the skeleton —
// the parallel write phase then changes neither file size nor metadata).
void h5_write_zero(hid_t loc, const std::string& name, hid_t htype,
                   const std::vector<hsize_t>& dims);

// Write data into an ALREADY existing dataset (opens then writes).
void h5_write_existing(hid_t loc, const std::string& name, hid_t htype, const void* data,
                       const std::vector<hsize_t>& dims);

// Hyperslab write into an existing dataset: start offsets along the first
// axis, local_dims is the block shape. Collective MPI-IO transfer
// (all ranks must call it on the same dataset).
void h5_write_slab(hid_t loc, const std::string& name, hid_t htype, const void* data, hsize_t start,
                   const std::vector<hsize_t>& local_dims);

// Hyperslab read (independent).
void h5_read_slab(hid_t loc, const std::string& name, hid_t htype, void* data, hsize_t start,
                  const std::vector<hsize_t>& local_dims);

// Read a whole dataset (the size is dictated by the caller).
void h5_read(hid_t loc, const std::string& name, hid_t htype, void* data,
             const std::vector<hsize_t>& dims);

// Read a dataset into a vector of the matching type (size checked).
std::vector<double> h5_read_d(hid_t loc, const std::string& name, size_t expect_elems);
std::vector<int32_t> h5_read_i(hid_t loc, const std::string& name, size_t expect_elems);
std::vector<int64_t> h5_read_l(hid_t loc, const std::string& name, size_t expect_elems);
std::vector<uint8_t> h5_read_u8(hid_t loc, const std::string& name, size_t expect_elems);

// Scalar attributes.
void h5_attr(hid_t loc, const std::string& name, hid_t htype, const void* v);
// Vector attribute of length n.
void h5_attr_vec(hid_t loc, const std::string& name, hid_t htype, const void* v, size_t n);
double h5_attr_d(hid_t loc, const std::string& name);
long long h5_attr_l(hid_t loc, const std::string& name);
int h5_attr_i(hid_t loc, const std::string& name);

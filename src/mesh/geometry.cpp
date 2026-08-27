#include "cfd/mesh/geometry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>

#include "cfd/mpi/log.hpp"

namespace cfd::mesh {

[[nodiscard]] double poly_cell_volume(CellType t, 
                                      const double* x, 
                                      const double* y, 
                                      const double* z) noexcept {
    // cast cell type
    const auto ti = static_cast<std::size_t>(t);

    // get number of cell faces
    const auto num_faces = static_cast<std::size_t>(kFacesPerType[ti]);

    double volume = 0.0;

    // loop over cell faces
    for (std::size_t f = 0; f < num_faces; ++f) {
        // get number of face nodes
        const auto nn = static_cast<std::size_t>(kFaceNodes[ti][f]);

        double S[3] = {0.0, 0.0, 0.0};
        double c_sum[3] = {0.0, 0.0, 0.0};

        // loop over face nodes
        for (std::size_t j = 0; j < nn; ++j) {
            const std::size_t next_j = (j + 1 == nn) ? 0 : (j + 1);

            // get node indices (local in cell nodes)
            const auto idx_a = static_cast<std::size_t>(kFaceTable[ti][f][j]);
            const auto idx_b = static_cast<std::size_t>(kFaceTable[ti][f][next_j]);

            const double ax = x[idx_a], ay = y[idx_a], az = z[idx_a];
            const double bx = x[idx_b], by = y[idx_b], bz = z[idx_b];

            // Cross product: a x b
            S[0] += ay * bz - az * by;
            S[1] += az * bx - ax * bz;
            S[2] += ax * by - ay * bx;

            c_sum[0] += ax;
            c_sum[1] += ay;
            c_sum[2] += az;
        } // end loop over face nodes

        // Divergence theorem: (1/3) * (S/2) · (c_sum/nn) = (S · c_sum) / (6 * nn)
        volume += (S[0] * c_sum[0] + S[1] * c_sum[1] + S[2] * c_sum[2]) / 
                  (6.0 * static_cast<double>(nn));
    } // end loop over cell faces

    return volume;
}

bool validate_face_tables() {
    // idial reference cells ("unit cells")
    struct Ref {
        CellType t;
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        double vol;
        const char* name;
    };

    const std::vector<Ref> refs = {
        {
            CellType::TET,
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
            1.0 / 6.0,
            "TET"
        },
        {
            CellType::PYRA,
            {0.0, 1.0, 1.0, 0.0, 0.5},
            {0.0, 0.0, 1.0, 1.0, 0.5},
            {0.0, 0.0, 0.0, 0.0, 1.0},
            1.0 / 3.0,
            "PYRA"
        },
        {
            CellType::PRISM,
            {0.0, 1.0, 0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
            0.5,
            "PRISM"
        },
        {
            CellType::HEXA,
            {0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0},
            {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
            1.0,
            "HEXA"
        },
    };

    bool ok = true;

    // loop over reference cell types
    for (const auto& r : refs) {
        const auto ti  = static_cast<std::size_t>(r.t);
        const auto npt = static_cast<std::size_t>(kNodesPerType[ti]);

        // Cell centroid computation
        double cc[3] = {0.0, 0.0, 0.0};

        // loop over cell nodes for centroid
        for (std::size_t i = 0; i < npt; ++i) {
            cc[0] += r.x[i];
            cc[1] += r.y[i];
            cc[2] += r.z[i];
        } // end loop over cell nodes for centroid
        
        const double inv_npt = 1.0 / static_cast<double>(npt);
        
        // loop over spatial dimensions for centroid normalization
        for (std::size_t d = 0; d < 3; ++d) {
            cc[d] *= inv_npt;
        } // end loop over spatial dimensions for centroid normalization

        // Volume check
        const double v = poly_cell_volume(r.t, r.x.data(), r.y.data(), r.z.data());
        if (std::abs(v - r.vol) > 1e-12 || v <= 0.0) {
            std::fprintf(stderr, "TABLES: %s volume %.12e != %.12e\n", r.name, v, r.vol);
            ok = false;
        }

        const auto num_faces = static_cast<std::size_t>(kFacesPerType[ti]);

        // loop over cell faces
        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto nn = static_cast<std::size_t>(kFaceNodes[ti][f]);

            double S[3] = {0.0, 0.0, 0.0};
            double cf[3] = {0.0, 0.0, 0.0};

            // loop over face nodes
            for (std::size_t j = 0; j < nn; ++j) {
                const std::size_t next_j = (j + 1 == nn) ? 0 : (j + 1);

                const auto idx_a = static_cast<std::size_t>(kFaceTable[ti][f][j]);
                const auto idx_b = static_cast<std::size_t>(kFaceTable[ti][f][next_j]);

                const double ax = r.x[idx_a], ay = r.y[idx_a], az = r.z[idx_a];
                const double bx = r.x[idx_b], by = r.y[idx_b], bz = r.z[idx_b];

                // Cross product: a x b
                S[0] += ay * bz - az * by;
                S[1] += az * bx - ax * bz;
                S[2] += ax * by - ay * bx;

                cf[0] += ax;
                cf[1] += ay;
                cf[2] += az;
            } // end loop over face nodes

            const double inv_nn = 1.0 / static_cast<double>(nn);
            // loop over spatial dimensions for face center normalization
            for (std::size_t d = 0; d < 3; ++d) {
                cf[d] *= inv_nn;
            } // end loop over spatial dimensions for face center normalization

            // Dot product between outer normal S and vector (cf - cc)
            const double dot = S[0] * (cf[0] - cc[0]) + 
                               S[1] * (cf[1] - cc[1]) + 
                               S[2] * (cf[2] - cc[2]);

            if (dot <= 0.0) {
                std::fprintf(stderr, "TABLES: %s face %zu inward normal (dot %.3e)\n", 
                             r.name, f, dot);
                ok = false;
            }
        } // end loop over cell faces

        // Permutation volume sign inversion check
        std::vector<double> flipped_x(npt);
        std::vector<double> flipped_y(npt);
        std::vector<double> flipped_z(npt);

        // loop over cell nodes for orientation flip
        for (std::size_t i = 0; i < npt; ++i) {
            const auto src_idx = static_cast<std::size_t>(kOrientationFlip[ti][i]);
            flipped_x[i] = r.x[src_idx];
            flipped_y[i] = r.y[src_idx];
            flipped_z[i] = r.z[src_idx];
        } // end loop over cell nodes for orientation flip

        const double vf = poly_cell_volume(r.t, flipped_x.data(), flipped_y.data(), flipped_z.data());
        if (vf >= 0.0) {
            std::fprintf(stderr, "TABLES: %s orientation flip did not invert volume sign (vol = %.6e)\n", 
                         r.name, vf);
            ok = false;
        }
    } // end loop over reference cell types

    return ok;
}

void compute_mesh_geometry(MeshPart& mp) {
    const auto n_cells_sz = static_cast<std::size_t>(mp.n_cells);
    const auto n_faces_sz = static_cast<std::size_t>(mp.n_faces);

    // -------------------------------------------------------------------------
    // Step 1: Pre-allocate SoA Geometric Arrays
    // -------------------------------------------------------------------------
    mp.cell_centroid_x.resize(n_cells_sz);
    mp.cell_centroid_y.resize(n_cells_sz);
    mp.cell_centroid_z.resize(n_cells_sz);
    mp.cell_volume.resize(n_cells_sz);

    mp.face_centroid_x.resize(n_faces_sz);
    mp.face_centroid_y.resize(n_faces_sz);
    mp.face_centroid_z.resize(n_faces_sz);
    mp.face_normal_x.resize(n_faces_sz);
    mp.face_normal_y.resize(n_faces_sz);
    mp.face_normal_z.resize(n_faces_sz);
    mp.face_area.resize(n_faces_sz);

    // -------------------------------------------------------------------------
    // Step 2: Compute Cell Metrics & Strict Positive Volume Validation
    // -------------------------------------------------------------------------
    double min_local_vol = std::numeric_limits<double>::max();
    double max_local_vol = -std::numeric_limits<double>::max();
    double total_local_vol = 0.0;

    double cell_x_buf[8];
    double cell_y_buf[8];
    double cell_z_buf[8];

    for (LocalIndex c = 0; c < mp.n_cells; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        const LocalIndex off_start = mp.cell_nodes_offsets[c_sz];
        const LocalIndex off_end   = mp.cell_nodes_offsets[c_sz + 1];
        const auto nnodes = static_cast<std::size_t>(off_end - off_start);
        const CellType type = mp.cell_type[c_sz];

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;

        for (std::size_t k = 0; k < nnodes; ++k) {
            const LocalIndex nid = mp.cell_nodes[static_cast<std::size_t>(off_start) + k];
            const auto nid_sz = static_cast<std::size_t>(nid);

            const double px = mp.node_x[nid_sz];
            const double py = mp.node_y[nid_sz];
            const double pz = mp.node_z[nid_sz];

            cell_x_buf[k] = px;
            cell_y_buf[k] = py;
            cell_z_buf[k] = pz;

            sum_x += px;
            sum_y += py;
            sum_z += pz;
        }

        const double inv_nn = 1.0 / static_cast<double>(nnodes);
        mp.cell_centroid_x[c_sz] = sum_x * inv_nn;
        mp.cell_centroid_y[c_sz] = sum_y * inv_nn;
        mp.cell_centroid_z[c_sz] = sum_z * inv_nn;

        const double vol = poly_cell_volume(type, cell_x_buf, cell_y_buf, cell_z_buf);

        // Strict positive volume assertion
        if (vol <= 1e-15) {
            std::stringstream ss;
            ss << "Degenerate/negative cell volume detected on Rank " << mp.rank
               << " (Local cell: " << c << ", Global GID: " << mp.cell_gid[c_sz]
               << ", Type: " << cell_type_name(type) << ", Volume: " << vol << ")";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }

        mp.cell_volume[c_sz] = vol;

        if (c < mp.n_own) {
            min_local_vol = std::min(min_local_vol, vol);
            max_local_vol = std::max(max_local_vol, vol);
            total_local_vol += vol;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Compute Face Metrics & Normal Vectors
    // -------------------------------------------------------------------------
    double min_local_area = std::numeric_limits<double>::max();
    double max_local_area = -std::numeric_limits<double>::max();

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const auto f_sz = static_cast<std::size_t>(f);
        const LocalIndex off_start = mp.face_nodes_offsets[f_sz];
        const LocalIndex off_end   = mp.face_nodes_offsets[f_sz + 1];
        const auto nnodes = static_cast<std::size_t>(off_end - off_start);

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;

        double fx[4];
        double fy[4];
        double fz[4];

        for (std::size_t k = 0; k < nnodes; ++k) {
            const LocalIndex nid = mp.face_nodes[static_cast<std::size_t>(off_start) + k];
            const auto nid_sz = static_cast<std::size_t>(nid);

            fx[k] = mp.node_x[nid_sz];
            fy[k] = mp.node_y[nid_sz];
            fz[k] = mp.node_z[nid_sz];

            sum_x += fx[k];
            sum_y += fy[k];
            sum_z += fz[k];
        }

        const double inv_nn = 1.0 / static_cast<double>(nnodes);
        const double fc_x = sum_x * inv_nn;
        const double fc_y = sum_y * inv_nn;
        const double fc_z = sum_z * inv_nn;

        mp.face_centroid_x[f_sz] = fc_x;
        mp.face_centroid_y[f_sz] = fc_y;
        mp.face_centroid_z[f_sz] = fc_z;

        // Compute unnormalized area vector (Right-hand rule outward normal)
        double Sx = 0.0;
        double Sy = 0.0;
        double Sz = 0.0;

        if (nnodes == 3) {
            // Triangle: S = 0.5 * (v1 - v0) x (v2 - v0)
            const double e1x = fx[1] - fx[0], e1y = fy[1] - fy[0], e1z = fz[1] - fz[0];
            const double e2x = fx[2] - fx[0], e2y = fy[2] - fy[0], e2z = fz[2] - fz[0];

            Sx = 0.5 * (e1y * e2z - e1z * e2y);
            Sy = 0.5 * (e1z * e2x - e1x * e2z);
            Sz = 0.5 * (e1x * e2y - e1y * e2x);
        } else if (nnodes == 4) {
            // Quad: S = 0.5 * (v2 - v0) x (v3 - v1) [Diagonal Cross Product]
            const double d1x = fx[2] - fx[0], d1y = fy[2] - fy[0], d1z = fz[2] - fz[0];
            const double d2x = fx[3] - fx[1], d2y = fy[3] - fy[1], d2z = fz[3] - fz[1];

            Sx = 0.5 * (d1y * d2z - d1z * d2y);
            Sy = 0.5 * (d1z * d2x - d1x * d2z);
            Sz = 0.5 * (d1x * d2y - d1y * d2x);
        } else {
            std::stringstream ss;
            ss << "Face " << f << " has unsupported node count: " << static_cast<int>(nnodes);
            std::string result = ss.str(); 
            mpi::fatal(MPI_COMM_WORLD, result);
        }

        const double area = std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz);

        if (area <= 1e-15) {
            std::stringstream ss;
            ss << "Degenerate zero-area face detected on Rank " << mp.rank
               << " (Face index: " << f << ", Area: " << area << ")";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }

        const double inv_area = 1.0 / area;
        const double nx = Sx * inv_area;
        const double ny = Sy * inv_area;
        const double nz = Sz * inv_area;

        mp.face_area[f_sz]     = area;
        mp.face_normal_x[f_sz] = nx;
        mp.face_normal_y[f_sz] = ny;
        mp.face_normal_z[f_sz] = nz;

        min_local_area = std::min(min_local_area, area);
        max_local_area = std::max(max_local_area, area);

        // ---------------------------------------------------------------------
        // Step 4: Verification of Outward Normal Alignment
        // Normal must point from face_owner outward towards face_neigh.
        // ---------------------------------------------------------------------
        const auto owner_sz = static_cast<std::size_t>(mp.face_owner[f_sz]);
        const double oc_x = mp.cell_centroid_x[owner_sz];
        const double oc_y = mp.cell_centroid_y[owner_sz];
        const double oc_z = mp.cell_centroid_z[owner_sz];

        const double d_vec_x = fc_x - oc_x;
        const double d_vec_y = fc_y - oc_y;
        const double d_vec_z = fc_z - oc_z;

        const double dot = nx * d_vec_x + ny * d_vec_y + nz * d_vec_z;
        if (dot <= 0.0) {
            std::stringstream ss;
            ss << "Normal orientation mismatch on Rank " << mp.rank
               << " for Face " << f << " (Owner cell: " << mp.face_owner[f_sz]
               << ", normal dot (fc - cc) = " << dot << " <= 0)";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Global Mesh Quality & Consistency Logging
    // -------------------------------------------------------------------------
    double min_glob_vol = 0.0, max_glob_vol = 0.0, total_glob_vol = 0.0;
    double min_glob_area = 0.0, max_glob_area = 0.0;

    MPI_Allreduce(&min_local_vol, &min_glob_vol, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_vol, &max_glob_vol, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&total_local_vol, &total_glob_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&min_local_area, &min_glob_area, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_area, &max_glob_area, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mp.rank == 0) {
        mpi::log_stat("INFO[Geometry computation]: Geometry verification passed successfully.");
        mpi::log_stat("      Total Domain Volume = %.8e", total_glob_vol);
        mpi::log_stat("      Cell Volumes : min = %.6e, max = %.6e", min_glob_vol, max_glob_vol);
        mpi::log_stat("      Face Areas   : min = %.6e, max = %.6e", min_glob_area, max_glob_area);
    }
}

} //namespace cfd::mesh
#include "cfd/mesh/geometry.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd/mpi/log.hpp"

double poly_cell_volume(CellType t, const double* pts) {
    const int ti = static_cast<int>(t);
    const int npt = kNodesPerType[ti];
    auto node = [&](int i) -> const double* { return pts + 3 * i; };
    double volume = 0.0;
    for (int f = 0; f < kFacesPerType[ti]; ++f) {
        const int nn = kFaceNodes[ti][f];
        // Area vector by fan: S = 1/2 sum (x_j x x_{j+1})
        double S[3] = {0, 0, 0};
        double c[3] = {0, 0, 0};
        for (int j = 0; j < nn; ++j) {
            const double* a = node(kFaceTable[ti][f][j]);
            const double* b = node(kFaceTable[ti][f][(j + 1) % nn]);
            S[0] += a[1] * b[2] - a[2] * b[1];
            S[1] += a[2] * b[0] - a[0] * b[2];
            S[2] += a[0] * b[1] - a[1] * b[0];
            for (int d = 0; d < 3; ++d) c[d] += a[d];
        }
        for (int d = 0; d < 3; ++d) c[d] /= nn;
        // S carries no 1/2 here, hence the divisor 6 = 1/(3*2)
        volume += (S[0] * c[0] + S[1] * c[1] + S[2] * c[2]) / 6.0;
    }
    (void)npt;
    return volume;
}

bool validate_face_tables() {
    struct Ref {
        CellType t;
        std::vector<double> pts;
        double vol;
        const char* name;
    };
    std::vector<Ref> refs = {
        {CellType::TET, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 1.0 / 6.0, "TET"},
        {CellType::PYRA, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1}, 1.0 / 3.0, "PYRA"},
        {CellType::PRISM, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1}, 0.5, "PRISM"},
        {CellType::HEXA,
         {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1},
         1.0,
         "HEXA"},
    };
    bool ok = true;
    for (const Ref& r : refs) {
        const int ti = static_cast<int>(r.t);
        const int npt = kNodesPerType[ti];
        // centroid
        double cc[3] = {0, 0, 0};
        for (int i = 0; i < npt; ++i)
            for (int d = 0; d < 3; ++d) cc[d] += r.pts[3 * i + d];
        for (int d = 0; d < 3; ++d) cc[d] /= npt;

        const double v = poly_cell_volume(r.t, r.pts.data());
        if (std::fabs(v - r.vol) > 1e-12 || v <= 0.0) {
            std::fprintf(stderr, "TABLES: %s volume %.6f != %.6f\n", r.name, v, r.vol);
            ok = false;
        }
        for (int f = 0; f < kFacesPerType[ti]; ++f) {
            const int nn = kFaceNodes[ti][f];
            double S[3] = {0, 0, 0}, cf[3] = {0, 0, 0};
            for (int j = 0; j < nn; ++j) {
                const double* a = &r.pts[3 * kFaceTable[ti][f][j]];
                const double* b = &r.pts[3 * kFaceTable[ti][f][(j + 1) % nn]];
                S[0] += a[1] * b[2] - a[2] * b[1];
                S[1] += a[2] * b[0] - a[0] * b[2];
                S[2] += a[0] * b[1] - a[1] * b[0];
                for (int d = 0; d < 3; ++d) cf[d] += a[d];
            }
            for (int d = 0; d < 3; ++d) cf[d] /= nn;
            const double dot =
                S[0] * (cf[0] - cc[0]) + S[1] * (cf[1] - cc[1]) + S[2] * (cf[2] - cc[2]);
            if (dot <= 0.0) {
                std::fprintf(stderr, "TABLES: %s face %d inward normal (dot %.3e)\n", r.name, f,
                             dot);
                ok = false;
            }
        }
        // The orientation permutation must yield a negative volume.
        std::vector<double> flipped(3 * npt);
        for (int i = 0; i < npt; ++i)
            for (int d = 0; d < 3; ++d) flipped[3 * i + d] = r.pts[3 * kOrientationFlip[ti][i] + d];
        const double vf = poly_cell_volume(r.t, flipped.data());
        if (vf >= 0.0) {
            std::fprintf(stderr, "TABLES: %s the flip does not change the volume sign\n", r.name);
            ok = false;
        }
    }
    return ok;
}

bool compute_geometry(MeshPart& mp) {
    bool ok = true;
    mp.cell_centroid.assign(static_cast<size_t>(mp.n_cells) * 3, 0.0);
    mp.cell_volume.assign(mp.n_cells, 0.0);
    std::vector<double> pts(24);
    for (int i = 0; i < mp.n_cells; ++i) {
        const int ti = mp.cell_type[i];
        const int npt = kNodesPerType[ti];
        for (int k = 0; k < npt; ++k) {
            const int32_t n = mp.cell_nodes[i * 8 + k];
            for (int d = 0; d < 3; ++d) {
                const double v = mp.node_xyz[3 * n + d];
                pts[3 * k + d] = v;
                mp.cell_centroid[3 * i + d] += v;
            }
        }
        for (int d = 0; d < 3; ++d) mp.cell_centroid[3 * i + d] /= npt;
        mp.cell_volume[i] = poly_cell_volume(static_cast<CellType>(ti), pts.data());
        if (mp.cell_volume[i] <= 0.0) {
            log_warn_rank(
                "cell gid %d: volume %.3e <= 0 (after the "
                "orientation fix)",
                mp.cell_gid[i], mp.cell_volume[i]);
            ok = false;
        }
    }

    mp.face_centroid.assign(static_cast<size_t>(mp.n_faces) * 3, 0.0);
    mp.face_normal.assign(static_cast<size_t>(mp.n_faces) * 3, 0.0);
    mp.face_area.assign(mp.n_faces, 0.0);
    for (int i = 0; i < mp.n_faces; ++i) {
        const int nn = mp.face_type[i] == static_cast<uint8_t>(CellType::TRI) ? 3 : 4;
        const int32_t* fn = &mp.face_nodes[i * 4];
        double S[3] = {0, 0, 0}, c[3] = {0, 0, 0};
        for (int j = 0; j < nn; ++j) {
            const double* a = &mp.node_xyz[3 * fn[j]];
            const double* b = &mp.node_xyz[3 * fn[(j + 1) % nn]];
            S[0] += a[1] * b[2] - a[2] * b[1];
            S[1] += a[2] * b[0] - a[0] * b[2];
            S[2] += a[0] * b[1] - a[1] * b[0];
            for (int d = 0; d < 3; ++d) c[d] += a[d];
        }
        for (int d = 0; d < 3; ++d) c[d] /= nn;
        const double area2 = std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
        mp.face_area[i] = 0.5 * area2;
        for (int d = 0; d < 3; ++d) {
            mp.face_centroid[3 * i + d] = c[d];
            mp.face_normal[3 * i + d] = 0.5 * S[d];
        }
        // The normal must point outward from the owner cell.
        const double* oc = &mp.cell_centroid[3 * mp.face_owner[i]];
        const double dot = mp.face_normal[3 * i] * (c[0] - oc[0]) +
                           mp.face_normal[3 * i + 1] * (c[1] - oc[1]) +
                           mp.face_normal[3 * i + 2] * (c[2] - oc[2]);
        if (dot <= 0.0) {
            log_warn_rank("face %d (cell gid %d): inward normal", i, mp.cell_gid[mp.face_owner[i]]);
            ok = false;
        }
    }
    return ok;
}

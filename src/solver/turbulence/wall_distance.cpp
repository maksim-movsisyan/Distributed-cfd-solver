#include "cfd/solver/turbulence/wall_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cfd/mpi/mpi_util.hpp"

namespace cfd::solver::turb {

namespace {
constexpr double kInf = 1.0e30;
constexpr double kMinDist = 1.0e-12;
} // anonymous namespace

void solve_wall_distance(const mesh::MeshPart& mp,
                         const gradient::VertexAdjacency& adj,
                         halo::HaloExchanger& halo,
                         const MPI_Comm comm,
                         const std::vector<bool>& wall_patch,
                         const int max_sweeps,
                         const double rel_tol,
                         double* CFD_RESTRICT dist) {
    static_cast<void>(comm);
    const std::size_t n_inner = static_cast<std::size_t>(mp.n_inner_faces);
    const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);
    const std::size_t n_cells = static_cast<std::size_t>(mp.n_cells);

    // 1. Initialize: +infinity everywhere, exact distances at wall-adjacent cells
    std::fill(dist, dist + n_cells, kInf);

    for (std::size_t f = n_inner; f < n_faces; ++f) {
        const std::size_t p = static_cast<std::size_t>(mp.face_patch[f]);
        if (p >= wall_patch.size() || !wall_patch[p]) {
            continue;
        }
        const std::size_t c0 = static_cast<std::size_t>(mp.face_owner[f]);
        const double ex = mp.face_centroid_x[f] - mp.cell_centroid_x[c0];
        const double ey = mp.face_centroid_y[f] - mp.cell_centroid_y[c0];
        const double ez = mp.face_centroid_z[f] - mp.cell_centroid_z[c0];
        const double d = std::sqrt(ex * ex + ey * ey + ez * ez);
        dist[c0] = std::min(dist[c0], d);
    }

    const LocalIndex* CFD_RESTRICT off = adj.offsets.data();
    const LocalIndex* CFD_RESTRICT nb = adj.cells.data();
    const double* CFD_RESTRICT adx = adj.dx.data();
    const double* CFD_RESTRICT ady = adj.dy.data();
    const double* CFD_RESTRICT adz = adj.dz.data();

    // 2. Alternating Gauss-Seidel sweep pairs with halo exchange + global
    //    convergence check (one collective per sweep pair, init-time only)
    double* dist_fields[] = {dist};
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        halo.exchange_once(dist_fields);

        double max_rel_change = 0.0;

        // Relaxation lambda shared by the forward and reverse passes
        const auto relax_cell = [&](const std::size_t c) {
            const double d0 = dist[c];
            double d_new = d0;

            for (LocalIndex j = off[c]; j < off[c + 1]; ++j) {
                const std::size_t js = static_cast<std::size_t>(j);
                const std::size_t n = static_cast<std::size_t>(nb[js]);
                const double cand = dist[n]
                                  + std::sqrt(adx[js] * adx[js]
                                            + ady[js] * ady[js]
                                            + adz[js] * adz[js]);
                d_new = (cand < d_new) ? cand : d_new;
            }

            if (d_new < d0) {
                const double ref = std::max(d0, kMinDist);
                const double rel = (d0 - d_new) / ref;
                max_rel_change = (rel > max_rel_change) ? rel : max_rel_change;
                dist[c] = d_new;
            }
        };

        for (std::size_t c = 0; c < static_cast<std::size_t>(mp.n_own); ++c) {
            relax_cell(c);
        }
        for (std::size_t c = static_cast<std::size_t>(mp.n_own); c-- > 0;) {
            relax_cell(c);
        }

        const double global_change = mpi::d_max(max_rel_change);
        if (global_change <= rel_tol) {
            break;
        }
    }
}

} // namespace cfd::solver::turb

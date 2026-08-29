#include "cfd/solver/gradient/gradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::solver::gradient {

VertexAdjacency build_vertex_adjacency(const mesh::MeshPart& mp) {
    const auto n_cells = static_cast<std::size_t>(mp.n_cells);
    const auto n_own   = static_cast<std::size_t>(mp.n_own);
    const auto n_nodes = static_cast<std::size_t>(mp.n_nodes);

    VertexAdjacency adj;
    adj.offsets.assign(n_own + 1, 0);

    if (n_own == 0 || n_nodes == 0) {
        return adj;
    }

    // 1. Build Node -> Cells CSR connectivity
    std::vector<LocalIndex> node_off(n_nodes + 1, 0);
    for (std::size_t c = 0; c < n_cells; ++c) {
        for (LocalIndex k = mp.cell_nodes_offsets[c]; k < mp.cell_nodes_offsets[c + 1]; ++k) {
            const auto n = static_cast<std::size_t>(mp.cell_nodes[static_cast<std::size_t>(k)]);
            ++node_off[n + 1];
        }
    }

    for (std::size_t n = 0; n < n_nodes; ++n) {
        node_off[n + 1] += node_off[n];
    }

    std::vector<LocalIndex> node_cells(static_cast<std::size_t>(node_off.back()));
    {
        std::vector<LocalIndex> cursor(node_off.begin(), node_off.end() - 1);
        for (std::size_t c = 0; c < n_cells; ++c) {
            for (LocalIndex k = mp.cell_nodes_offsets[c]; k < mp.cell_nodes_offsets[c + 1]; ++k) {
                const auto n = static_cast<std::size_t>(mp.cell_nodes[static_cast<std::size_t>(k)]);
                node_cells[static_cast<std::size_t>(cursor[n]++)] = static_cast<LocalIndex>(c);
            }
        }
    }

    // 2. Pass 1: Count unique vertex-sharing neighbours per owned cell (Zero Allocations)
    std::vector<LocalIndex> stamp(n_cells, -1);
    for (std::size_t c = 0; c < n_own; ++c) {
        const auto tag = static_cast<LocalIndex>(c);
        LocalIndex count = 0;

        for (LocalIndex k = mp.cell_nodes_offsets[c]; k < mp.cell_nodes_offsets[c + 1]; ++k) {
            const auto n = static_cast<std::size_t>(mp.cell_nodes[static_cast<std::size_t>(k)]);
            for (LocalIndex j = node_off[n]; j < node_off[n + 1]; ++j) {
                const LocalIndex cand = node_cells[static_cast<std::size_t>(j)];
                const auto cs = static_cast<std::size_t>(cand);

                if (cand == static_cast<LocalIndex>(c) || stamp[cs] == tag) {
                    continue;
                }
                stamp[cs] = tag;
                ++count;
            }
        }
        adj.offsets[c + 1] = adj.offsets[c] + count;
    }

    // 3. Allocate flat CSR buffers
    const auto total = static_cast<std::size_t>(adj.offsets.back());
    adj.cells.resize(total);
    adj.dx.resize(total);
    adj.dy.resize(total);
    adj.dz.resize(total);
    adj.w.resize(total);

    // Reset stamp array cleanly for Pass 2
    std::fill(stamp.begin(), stamp.end(), -1);

    // 4. Pass 2: Populate neighbours, sort for cache-locality, compute metrics
    std::vector<LocalIndex> tmp;
    tmp.reserve(64);

    for (std::size_t c = 0; c < n_own; ++c) {
        const auto tag = static_cast<LocalIndex>(c);
        tmp.clear();

        for (LocalIndex k = mp.cell_nodes_offsets[c]; k < mp.cell_nodes_offsets[c + 1]; ++k) {
            const auto n = static_cast<std::size_t>(mp.cell_nodes[static_cast<std::size_t>(k)]);
            for (LocalIndex j = node_off[n]; j < node_off[n + 1]; ++j) {
                const LocalIndex cand = node_cells[static_cast<std::size_t>(j)];
                const auto cs = static_cast<std::size_t>(cand);

                if (cand == static_cast<LocalIndex>(c) || stamp[cs] == tag) {
                    continue;
                }
                stamp[cs] = tag;
                tmp.push_back(cand);
            }
        }

        // Sorting neighbours ensures monotonic memory access to field states
        std::sort(tmp.begin(), tmp.end());

        const double cx = mp.cell_centroid_x[c];
        const double cy = mp.cell_centroid_y[c];
        const double cz = mp.cell_centroid_z[c];

        for (LocalIndex j = adj.offsets[c]; j < adj.offsets[c + 1]; ++j) {
            const auto js = static_cast<std::size_t>(tmp[static_cast<std::size_t>(j - adj.offsets[c])]);
            adj.cells[static_cast<std::size_t>(j)] = static_cast<LocalIndex>(js);

            const double ex = mp.cell_centroid_x[js] - cx;
            const double ey = mp.cell_centroid_y[js] - cy;
            const double ez = mp.cell_centroid_z[js] - cz;

            adj.dx[static_cast<std::size_t>(j)] = ex;
            adj.dy[static_cast<std::size_t>(j)] = ey;
            adj.dz[static_cast<std::size_t>(j)] = ez;

            const double dist_sq = ex * ex + ey * ey + ez * ez;
            adj.w[static_cast<std::size_t>(j)]  = 1.0 / std::sqrt(std::max(dist_sq, 1.0e-30));
        }
    }

    return adj;
}

} // namespace cfd::solver::gradient
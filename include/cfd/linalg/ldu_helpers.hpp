#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"

namespace cfd::linalg::detail {

/**
 * @brief Builds LDU graph topology from cell-to-cell dual graph.
 * Separates internal edges (c < nb < n_own) from interface edges (c < n_own <= nb).
 */
inline void ldu_assemble_from_dual_graph(
    const std::vector<LocalIndex>& dual_graph_off,
    const std::vector<LocalIndex>& dual_graph_val,
    std::vector<LocalIndex>& owner_start,
    std::vector<LocalIndex>& owner,
    std::vector<LocalIndex>& neigh,
    std::size_t& n_internal_edges) {

    const std::size_t n_own = dual_graph_off.size() - 1;

    owner_start.resize(n_own + 1, 0);

    std::vector<std::pair<LocalIndex, LocalIndex>> internal_edges;
    std::vector<std::pair<LocalIndex, LocalIndex>> interface_edges;

    // Dual graph traversal: owner c is strictly local
    for (std::size_t c = 0; c < n_own; ++c) {
        owner_start[c] = static_cast<LocalIndex>(internal_edges.size());

        const std::size_t ptr_start = static_cast<std::size_t>(dual_graph_off[c]);
        const std::size_t ptr_end   = static_cast<std::size_t>(dual_graph_off[c + 1]);

        for (std::size_t e = ptr_start; e < ptr_end; ++e) {
            const auto nb = dual_graph_val[e];
            const auto u_node = static_cast<LocalIndex>(c);

            if (static_cast<std::size_t>(nb) < n_own) {
                // Internal face: preserve u < v convention
                if (u_node < nb) {
                    internal_edges.emplace_back(u_node, nb);
                }
            } else {
                // Inter-processor interface face: local owner -> halo ghost
                interface_edges.emplace_back(u_node, nb);
            }
        }
    }
    owner_start[n_own] = static_cast<LocalIndex>(internal_edges.size());
    n_internal_edges   = internal_edges.size();

    const std::size_t n_total = n_internal_edges + interface_edges.size();
    owner.resize(n_total);
    neigh.resize(n_total);

    // Pack internal edges first [0, n_internal_edges)
    for (std::size_t k = 0; k < n_internal_edges; ++k) {
        owner[k] = internal_edges[k].first;
        neigh[k] = internal_edges[k].second;
    }

    // Append interface edges [n_internal_edges, n_total)
    const std::size_t n_interf = interface_edges.size();
    for (std::size_t k = 0; k < n_interf; ++k) {
        owner[n_internal_edges + k] = interface_edges[k].first;
        neigh[n_internal_edges + k] = interface_edges[k].second;
    }
}

/**
 * @brief Directly builds LDU graph from mesh face arrays.
 */
inline void ldu_assemble_from_faces(
    std::size_t n_own,
    const std::vector<LocalIndex>& face_owner,
    const std::vector<LocalIndex>& face_neigh,
    std::size_t n_internal_faces,
    std::vector<LocalIndex>& owner_start,
    std::vector<LocalIndex>& owner,
    std::vector<LocalIndex>& neigh,
    std::size_t& n_internal_edges) {

    owner_start.assign(n_own + 1, 0);

    owner = face_owner;
    neigh = face_neigh;
    n_internal_edges = n_internal_faces;

    // Count internal edges owned by each cell
    for (std::size_t f = 0; f < n_internal_faces; ++f) {
        const auto u = static_cast<std::size_t>(owner[f]);
        if (u < n_own) {
            ++owner_start[u + 1];
        }
    }
    for (std::size_t i = 0; i < n_own; ++i) {
        owner_start[i + 1] += owner_start[i];
    }
}

/**
 * @brief High-performance scalar SpMV for LDU matrices: y = alpha * A * x + beta * y
 */
inline void ldu_spmv(LocalIndex n_rows,
                     std::size_t n_internal,
                     std::size_t n_total,
                     const LocalIndex* CFD_RESTRICT owner,
                     const LocalIndex* CFD_RESTRICT neigh,
                     const double* CFD_RESTRICT diag,
                     const double* CFD_RESTRICT upper,
                     const double* CFD_RESTRICT lower,
                     const double* CFD_RESTRICT x,
                     double* CFD_RESTRICT y,
                     const double alpha,
                     const double beta) {

    const auto n_sz = static_cast<std::size_t>(n_rows);

    // 0. Short-circuit for alpha == 0
    if (alpha == 0.0) {
        if (beta == 0.0) {
            std::fill_n(y, n_sz, 0.0);
        } else if (beta != 1.0) {
            for (std::size_t i = 0; i < n_sz; ++i) y[i] *= beta;
        }
        return;
    }

    // 1. Diagonal contribution & beta scaling
    if (beta == 0.0) {
        if (alpha == 1.0) {
            for (std::size_t i = 0; i < n_sz; ++i) y[i] = diag[i] * x[i];
        } else {
            for (std::size_t i = 0; i < n_sz; ++i) y[i] = alpha * diag[i] * x[i];
        }
    } else if (beta == 1.0) {
        if (alpha == 1.0) {
            for (std::size_t i = 0; i < n_sz; ++i) y[i] += diag[i] * x[i];
        } else {
            for (std::size_t i = 0; i < n_sz; ++i) y[i] += alpha * diag[i] * x[i];
        }
    } else {
        for (std::size_t i = 0; i < n_sz; ++i) {
            y[i] = alpha * diag[i] * x[i] + beta * y[i];
        }
    }

    // 2. Internal edges traversal (both owner and neighbor are strictly local)
    if (alpha == 1.0) {
        for (std::size_t k = 0; k < n_internal; ++k) {
            const auto u = static_cast<std::size_t>(owner[k]);
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[u] += upper[k] * x[v];
            y[v] += lower[k] * x[u];
        }
    } else {
        for (std::size_t k = 0; k < n_internal; ++k) {
            const auto u = static_cast<std::size_t>(owner[k]);
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[u] += alpha * upper[k] * x[v];
            y[v] += alpha * lower[k] * x[u];
        }
    }

    // 3. Interface edges (owner is local, neighbor is halo ghost)
    if (alpha == 1.0) {
        for (std::size_t k = n_internal; k < n_total; ++k) {
            const auto u = static_cast<std::size_t>(owner[k]);
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[u] += upper[k] * x[v];
        }
    } else {
        for (std::size_t k = n_internal; k < n_total; ++k) {
            const auto u = static_cast<std::size_t>(owner[k]);
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[u] += alpha * upper[k] * x[v];
        }
    }
}

/**
 * @brief Symmetric Gauss-Seidel (SGS) preconditioner sweep on the internal LDU system.
 * Solves: (D + U) D^{-1} (D + L) z = r
 */
inline void ldu_sgs_sweep(
    LocalIndex n_rows,
    const LocalIndex* CFD_RESTRICT owner_start,
    const LocalIndex* CFD_RESTRICT neigh,
    const double* CFD_RESTRICT diag,
    const double* CFD_RESTRICT upper,
    const double* CFD_RESTRICT lower,
    const double* CFD_RESTRICT r,
    double* CFD_RESTRICT z) {

    const auto n = static_cast<std::size_t>(n_rows);

    static thread_local std::vector<double> work_buf;
    if (work_buf.size() < n) {
        work_buf.resize(n);
    }
    double* CFD_RESTRICT y = work_buf.data();
    std::copy_n(r, n, y);

    // Forward Sweep: (D + L) z* = r (Push strategy)
    for (std::size_t i = 0; i < n; ++i) {
        const double z_star_i = y[i] / diag[i];
        z[i] = z_star_i;

        const auto k_end = static_cast<std::size_t>(owner_start[i + 1]);
        for (std::size_t k = static_cast<std::size_t>(owner_start[i]); k < k_end; ++k) {
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[v] -= lower[k] * z_star_i;
        }
    }

    // Backward Sweep: (D + U) z = D z* (Pull strategy)
    for (std::size_t i = n; i-- > 0;) {
        double sum_u = 0.0;
        const auto k_end = static_cast<std::size_t>(owner_start[i + 1]);
        for (std::size_t k = static_cast<std::size_t>(owner_start[i]); k < k_end; ++k) {
            const auto v = static_cast<std::size_t>(neigh[k]);
            sum_u += upper[k] * z[v];
        }
        z[i] -= sum_u / diag[i];
    }
}


/**
 * @brief Forward sweep for LDU: (D + L) z* = r - U * z_old - U_interf * z_ghost
 */
inline void ldu_sgs_forward(
    LocalIndex n_rows,
    std::size_t n_int,
    std::size_t n_total,
    const LocalIndex* CFD_RESTRICT owner_start,
    const LocalIndex* CFD_RESTRICT owner,
    const LocalIndex* CFD_RESTRICT neigh,
    const double* CFD_RESTRICT inv_diag,
    const double* CFD_RESTRICT upper,
    const double* CFD_RESTRICT lower,
    const double* CFD_RESTRICT r,
    double* CFD_RESTRICT z,
    double* CFD_RESTRICT y,
    bool is_first_sweep) {

    const auto n = static_cast<std::size_t>(n_rows);

    std::copy_n(r, n, y);

    if (!is_first_sweep) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto k_end = static_cast<std::size_t>(owner_start[i + 1]);
            for (std::size_t k = static_cast<std::size_t>(owner_start[i]); k < k_end; ++k) {
                const auto v = static_cast<std::size_t>(neigh[k]);
                y[i] -= upper[k] * z[v];
            }
        }
        for (std::size_t k = n_int; k < n_total; ++k) {
            const auto u = static_cast<std::size_t>(owner[k]);
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[u] -= upper[k] * z[v];
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double zi = y[i] * inv_diag[i];
        z[i] = zi;

        const auto k_end = static_cast<std::size_t>(owner_start[i + 1]);
        for (std::size_t k = static_cast<std::size_t>(owner_start[i]); k < k_end; ++k) {
            const auto v = static_cast<std::size_t>(neigh[k]);
            y[v] -= lower[k] * zi;
        }
    }
}

/**
 * @brief Backward sweep for LDU: (D + U) z = r - L * z* - U_interf * z_ghost
 */
inline void ldu_sgs_backward(
    LocalIndex n_rows,
    std::size_t n_int,
    std::size_t n_total,
    const LocalIndex* CFD_RESTRICT owner_start,
    const LocalIndex* CFD_RESTRICT owner,
    const LocalIndex* CFD_RESTRICT neigh,
    const double* CFD_RESTRICT inv_diag,
    const double* CFD_RESTRICT upper,
    const double* CFD_RESTRICT lower,
    const double* CFD_RESTRICT r,
    double* CFD_RESTRICT z,
    double* CFD_RESTRICT y) {

    const auto n = static_cast<std::size_t>(n_rows);

    std::copy_n(r, n, y);

    for (std::size_t k = 0; k < n_int; ++k) {
        const auto u = static_cast<std::size_t>(owner[k]);
        const auto v = static_cast<std::size_t>(neigh[k]);
        y[v] -= lower[k] * z[u];
    }

    for (std::size_t k = n_int; k < n_total; ++k) {
        const auto u = static_cast<std::size_t>(owner[k]);
        const auto v = static_cast<std::size_t>(neigh[k]);
        y[u] -= upper[k] * z[v];
    }

    for (std::size_t i = n; i-- > 0;) {
        double s = y[i];
        const auto k_end = static_cast<std::size_t>(owner_start[i + 1]);
        for (std::size_t k = static_cast<std::size_t>(owner_start[i]); k < k_end; ++k) {
            const auto v = static_cast<std::size_t>(neigh[k]);
            s -= upper[k] * z[v];
        }
        z[i] = s * inv_diag[i];
    }
}

} // namespace cfd::linalg::detail
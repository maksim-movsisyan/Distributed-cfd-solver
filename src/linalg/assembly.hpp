// COO -> local CSR assembly shared by CsrMatrix and BsrMatrix.
// `values_per_edge` doubles live on each graph edge: 1 for CSR, bs*bs for BSR.
#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "cfd/linalg/types.hpp"
#include "cfd/linalg/vector_layout.hpp"

namespace cfd::linalg::detail {

/**
 * @brief Sorts the staged COO triplets by (row, col), merges duplicates by
 * summation and emits a CSR structure with LOCAL column slots, plus the
 * diagonal slot per row (`diag_idx`, kInvalidLocalIndex when absent).
 *
 * The layout must already know the ghosts (VectorLayout::setGhosts) — columns
 * are localized through it, so the output is directly usable by SpMV kernels.
 * Note: entries are sorted by GLOBAL column id; the mapped local slots are not
 * necessarily monotone within a row (ghost slots may precede owned ones), so
 * per-row lookups must scan rather than binary-search.
 * Rows must belong to this rank; failures abort.
 */
inline void assemble_coo(const VectorLayout& layout, std::size_t values_per_edge,
                         const std::vector<GlobalIndex>& coo_rows,
                         const std::vector<GlobalIndex>& coo_cols,
                         const std::vector<double>& coo_vals, std::vector<LocalIndex>& row_ptr,
                         std::vector<LocalIndex>& cols, std::vector<double>& values,
                         std::vector<LocalIndex>& diag_idx) {
    const MPI_Comm comm = layout.comm();
    const LocalIndex n_local = layout.localSize();
    const std::size_t n = coo_rows.size();
    const std::size_t n1 = static_cast<std::size_t>(n_local) + 1;

    row_ptr.assign(n1, 0);
    cols.clear();
    values.clear();
    if (n == 0) return;

    check(coo_cols.size() == n && coo_vals.size() == n * values_per_edge, comm,
          "assemble: malformed COO staging arrays");

    std::vector<LocalIndex> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&coo_rows, &coo_cols](LocalIndex a, LocalIndex b) {
        if (coo_rows[static_cast<std::size_t>(a)] != coo_rows[static_cast<std::size_t>(b)]) {
            return coo_rows[static_cast<std::size_t>(a)] < coo_rows[static_cast<std::size_t>(b)];
        }
        return coo_cols[static_cast<std::size_t>(a)] < coo_cols[static_cast<std::size_t>(b)];
    });

    const GlobalIndex begin = layout.localBegin();
    const GlobalIndex end = begin + n_local;

    // Pass 1: count unique edges per row (row histogram into row_ptr[1..]).
    std::size_t n_unique = 0;
    bool have_prev = false;
    GlobalIndex prev_row = 0, prev_col = 0;
    for (std::size_t e = 0; e < n; ++e) {
        const std::size_t idx = static_cast<std::size_t>(order[e]);
        const GlobalIndex r = coo_rows[idx];
        const GlobalIndex c = coo_cols[idx];
        check(r >= begin && r < end, comm, "assemble: COO row not owned by this rank");
        const bool is_new = !have_prev || r != prev_row || c != prev_col;
        if (is_new) {
            ++row_ptr[static_cast<std::size_t>(r - begin) + 1];
            ++n_unique;
            prev_row = r;
            prev_col = c;
            have_prev = true;
        }
    }
    check(n_unique <= static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()), comm,
          "assemble: nonzeros exceed the local index range");
    for (std::size_t i = 1; i < n1; ++i) row_ptr[i] += row_ptr[i - 1];

    // Pass 2: emit local columns and summed values (edges of one row are
    // consecutive, so slots run in lockstep with the sorted order); the
    // diagonal slot falls out of the emission for free.
    cols.resize(n_unique);
    values.assign(n_unique * values_per_edge, 0.0);
    diag_idx.assign(static_cast<std::size_t>(n_local), kInvalidLocalIndex);
    std::size_t slot = 0;
    have_prev = false;
    for (std::size_t e = 0; e < n; ++e) {
        const std::size_t idx = static_cast<std::size_t>(order[e]);
        const GlobalIndex r = coo_rows[idx];
        const GlobalIndex c = coo_cols[idx];
        if (!have_prev || r != prev_row || c != prev_col) {
            const LocalIndex lrow = static_cast<LocalIndex>(r - begin);
            const LocalIndex lcol = layout.localIndex(c);
            check(lcol != kInvalidLocalIndex, comm, "assemble: column not owned nor ghost");
            cols[slot] = lcol;
            if (lcol == lrow) diag_idx[static_cast<std::size_t>(lrow)] = static_cast<LocalIndex>(slot);
            prev_row = r;
            prev_col = c;
            have_prev = true;
            ++slot;
        }
        const double* src = coo_vals.data() + idx * values_per_edge;
        double* dst = values.data() + (slot - 1) * values_per_edge;
        for (std::size_t v = 0; v < values_per_edge; ++v) dst[v] += src[v];
    }
}

/**
 * @brief Slot of (local_row, local_col) inside an assembled structure, for
 * add-only value reassembly; kInvalidLocalIndex when not present.
 * Linear scan: local column slots are not sorted within a row.
 */
inline LocalIndex find_slot(const std::vector<LocalIndex>& row_ptr,
                            const std::vector<LocalIndex>& cols, LocalIndex local_row,
                            LocalIndex local_col) {
    for (LocalIndex k = row_ptr[static_cast<std::size_t>(local_row)];
         k < row_ptr[static_cast<std::size_t>(local_row) + 1]; ++k) {
        if (cols[static_cast<std::size_t>(k)] == local_col) return k;
    }
    return kInvalidLocalIndex;
}

}  // namespace cfd::linalg::detail

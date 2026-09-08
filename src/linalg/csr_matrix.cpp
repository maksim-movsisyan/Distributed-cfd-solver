#include "cfd/linalg/csr_matrix.hpp"
#include <cstddef>
#include <algorithm>
#include <iterator>

#include "cfd/core/types.hpp"
#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

void CsrMatrix::apply(const Vector& x, Vector& y, double alpha, double beta) const {
    check(assembled_, layout_.comm(), "CsrMatrix::apply: matrix not assembled");
    const LocalIndex n_rows = layout_.localSize();
    if (n_rows == 0) return;

    const LocalIndex* CFD_RESTRICT r_ptr = row_ptr_.data();
    const LocalIndex* CFD_RESTRICT c_idx = cols_.data();
    const double*     CFD_RESTRICT val   = values_.data();
    
    const double*     CFD_RESTRICT xv    = x.data();
    double*           CFD_RESTRICT yv    = y.data();

    if (alpha == 1.0 && beta == 0.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double sum = 0.0;
            const LocalIndex row_start = r_ptr[i];
            const LocalIndex row_end   = r_ptr[i + 1];

            for (LocalIndex j = row_start; j < row_end; ++j) {
                sum += val[j] * xv[c_idx[j]];
            }
            yv[i] = sum;
        }
    } else if (beta == 1.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double sum = 0.0;
            const LocalIndex row_start = r_ptr[i];
            const LocalIndex row_end   = r_ptr[i + 1];

            for (LocalIndex j = row_start; j < row_end; ++j) {
                sum += val[j] * xv[c_idx[j]];
            }
            yv[i] += alpha * sum;
        }
    } else if (beta != 0.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double sum = 0.0;
            const LocalIndex row_start = r_ptr[i];
            const LocalIndex row_end   = r_ptr[i + 1];

            for (LocalIndex j = row_start; j < row_end; ++j) {
                sum += val[j] * xv[c_idx[j]];
            }
            yv[i] = alpha * sum + beta * yv[i];
        }
    } else {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double sum = 0.0;
            const LocalIndex row_start = r_ptr[i];
            const LocalIndex row_end   = r_ptr[i + 1];

            for (LocalIndex j = row_start; j < row_end; ++j) {
                sum += val[j] * xv[c_idx[j]];
            }
            yv[i] = alpha * sum;
        }
    }
}

void CsrMatrix::assebmle(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                         const std::vector<LocalIndex>& dual_graph_off,
                         const std::vector<LocalIndex>& dual_graph_val) {
    // 0. get number of graph vertices (matrix rows == owned cells)
    const std::size_t n_own = dual_graph_off.size() - 1;
    check(layout_.localSize() == static_cast<LocalIndex>(n_own), layout_.comm(),
          "CsrMatrix::assemble: layout local size does not match graph row count");

    row_ptr_.resize(n_own + 1, 0);
    diag_idx_.resize(n_own);
    
    // 1. counting number of cols per each row (== 1 + number of neighbors)
    // loop over all own graph vertices
    for (std::size_t c = 0; c < n_own; ++c) { 
        const LocalIndex ncols = 1 + (dual_graph_off[c + 1] - dual_graph_off[c]);
        row_ptr_[c + 1] = row_ptr_[c] + ncols;
    }

    // 2. fill column indices
    const std::size_t nnz = static_cast<std::size_t>(row_ptr_[n_own]);
    cols_.resize(nnz);
    values_.resize(nnz, 0.0);

    layout_.setGhosts(sorted_unique_ghost_gids);

    std::size_t global_count = 0;

    // loop over all own graph vertices
    for (std::size_t c = 0; c < n_own; ++c) {
        const std::size_t row_start_pos = global_count;
        
        cols_[static_cast<std::size_t>(row_ptr_[c]++)] = static_cast<LocalIndex>(c);
        std::size_t count = 1;

        const std::size_t ptr1 = static_cast<std::size_t>(dual_graph_off[c]);
        const std::size_t ptr2 = static_cast<std::size_t>(dual_graph_off[c + 1]);
        
        for (std::size_t e = ptr1; e < ptr2; ++e) {
            cols_[static_cast<std::size_t>(row_ptr_[c]++)] = dual_graph_val[e];
            ++count;
        }

        auto it_beg = cols_.begin() + static_cast<std::ptrdiff_t>(row_start_pos);
        auto it_end = it_beg + static_cast<std::ptrdiff_t>(count);
        std::sort(it_beg, it_end);

        auto it = std::lower_bound(it_beg, it_end, static_cast<LocalIndex>(c));
        diag_idx_[c] = static_cast<LocalIndex>(row_start_pos) + static_cast<LocalIndex>(std::distance(it_beg, it));

        global_count += count;
    }
    
    // 3. restore row pointers
    for (std::size_t c = n_own; c > 0; --c) { 
        row_ptr_[c] = row_ptr_[c - 1];
    }
    row_ptr_[0] = 0;

    assembled_ = true;
}

}  // namespace cfd::linalg

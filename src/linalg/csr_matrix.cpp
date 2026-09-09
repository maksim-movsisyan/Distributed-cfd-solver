#include "cfd/linalg/csr_matrix.hpp"
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/linalg/types.hpp"
#include "cfd/linalg/bsr_helpers.hpp"

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

void CsrMatrix::assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                         const std::vector<LocalIndex>& dual_graph_off,
                         const std::vector<LocalIndex>& dual_graph_val) {
    check(!assembled_, layout_.comm(), "CsrMatrix::assemble: matrix is already assembled");

    const std::size_t n_own = dual_graph_off.size() - 1;
    check(layout_.localSize() == static_cast<LocalIndex>(n_own), layout_.comm(),
          "CsrMatrix::assemble: layout local size does not match graph row count");

    detail::bsr_assemble(dual_graph_off, dual_graph_val,
                         row_ptr_, diag_idx_, cols_);                        
    
    const std::size_t nnz = static_cast<std::size_t>(row_ptr_[n_own]);
    values_.assign(nnz, 0.0);

    layout_.setGhosts(sorted_unique_ghost_gids);

    assembled_ = true;
}

}  // namespace cfd::linalg

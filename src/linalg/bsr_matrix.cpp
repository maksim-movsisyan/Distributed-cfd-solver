#include "cfd/linalg/bsr_matrix.hpp"

#include <cstring>

#include "cfd/linalg/types.hpp"
#include "cfd/linalg/bsr_helpers.hpp"

namespace cfd::linalg {

void BsrMatrix::apply(const Vector& x, Vector& y, double alpha, double beta) const {
    check(assembled_, layout_.comm(), "BsrMatrix::apply: matrix not assembled");
    check(x.blockSize() == bs_ && y.blockSize() == bs_, layout_.comm(),
          "BsrMatrix::apply: vector block size mismatch");
    check(x.layout().compatibleWith(layout_) && y.layout().compatibleWith(layout_),
          layout_.comm(), "BsrMatrix::apply: incompatible vector layout");
    check(&x != &y, layout_.comm(), "BsrMatrix::apply: x and y must be distinct");

    detail::bsr_spmv_dispatch(layout_.localSize(), bs_, row_ptr_.data(), cols_.data(),
                              values_.data(), x.data(), y.data(), alpha, beta);
}

void BsrMatrix::assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                         const std::vector<LocalIndex>& dual_graph_off,
                         const std::vector<LocalIndex>& dual_graph_val) {
    check(!assembled_, layout_.comm(), "BsrMatri::assemble: matrix is already assembled");

    const std::size_t n_own = dual_graph_off.size() - 1;
    check(layout_.localSize() == static_cast<LocalIndex>(n_own), layout_.comm(),
          "BsrMatrix::assemble: layout local size does not match graph row count");

    detail::bsr_assemble(dual_graph_off, dual_graph_val,
                         row_ptr_, diag_idx_, cols_);                        
    
    const std::size_t nnz = static_cast<std::size_t>(row_ptr_[n_own]);
    values_.assign(nnz * static_cast<std::size_t>(blockSize() * blockSize()), 0.0);

    layout_.setGhosts(sorted_unique_ghost_gids);

    assembled_ = true;
}

}  // namespace cfd::linalg

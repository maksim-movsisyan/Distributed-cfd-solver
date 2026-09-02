#include "cfd/linalg/bsr_matrix.hpp"

#include <algorithm>
#include <cstring>

#include "assembly.hpp"
#include "bsr_kernels.hpp"
#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

BsrMatrix::BsrMatrix(MPI_Comm comm, GlobalIndex n_global_block_rows, LocalIndex n_local_block_rows,
                     int block_size)
    : layout_(comm, n_global_block_rows, n_local_block_rows), bs_(block_size) {
    check(block_size > 0, layout_.comm(), "BsrMatrix: block size must be positive");
}

void BsrMatrix::reserve(std::size_t n_blocks_hint) {
    const std::size_t vpe = static_cast<std::size_t>(bs_) * static_cast<std::size_t>(bs_);
    coo_rows_.reserve(n_blocks_hint);
    coo_cols_.reserve(n_blocks_hint);
    coo_vals_.reserve(n_blocks_hint * vpe);
}

void BsrMatrix::addBlock(GlobalIndex block_row, GlobalIndex block_col, const double* block_values) {
    // Staging is always allowed: assemble() consumes the entries into a new
    // structure, assembleValues() adds them into the existing one.
    check(block_row >= layout_.localBegin() &&
              block_row < layout_.localBegin() + layout_.localSize(),
          layout_.comm(), "BsrMatrix::addBlock: row not owned by this rank");
    check(block_col >= 0 && block_col < layout_.globalSize(), layout_.comm(),
          "BsrMatrix::addBlock: column outside the global range");
    const std::size_t vpe = static_cast<std::size_t>(bs_) * static_cast<std::size_t>(bs_);
    coo_rows_.push_back(block_row);
    coo_cols_.push_back(block_col);
    coo_vals_.insert(coo_vals_.end(), block_values, block_values + vpe);
}

void BsrMatrix::assemble() {
    check(!assembled_, layout_.comm(), "BsrMatrix::assemble: already assembled");

    std::vector<GlobalIndex> ghosts;
    const GlobalIndex begin = layout_.localBegin();
    const GlobalIndex end = begin + layout_.localSize();
    for (GlobalIndex c : coo_cols_) {
        if (c < begin || c >= end) ghosts.push_back(c);
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    layout_.setGhosts(ghosts);  // collective

    const std::size_t vpe = static_cast<std::size_t>(bs_) * static_cast<std::size_t>(bs_);
    detail::assemble_coo(layout_, vpe, coo_rows_, coo_cols_, coo_vals_, row_ptr_, cols_, values_,
                         diag_idx_);

    coo_rows_.clear();
    coo_cols_.clear();
    coo_vals_.clear();
    coo_rows_.shrink_to_fit();
    coo_cols_.shrink_to_fit();
    coo_vals_.shrink_to_fit();
    assembled_ = true;
}

void BsrMatrix::assembleValues() {
    check(assembled_, layout_.comm(),
          "BsrMatrix::assembleValues: structure unknown, call assemble() first");
    const GlobalIndex begin = layout_.localBegin();
    const std::size_t n = coo_rows_.size();
    const std::size_t vpe = static_cast<std::size_t>(bs_) * static_cast<std::size_t>(bs_);
    check(coo_vals_.size() == n * vpe, layout_.comm(),
          "BsrMatrix::assembleValues: malformed COO staging arrays");
    for (std::size_t e = 0; e < n; ++e) {
        const GlobalIndex r = coo_rows_[e];
        check(r >= begin && r < begin + layout_.localSize(), layout_.comm(),
              "BsrMatrix::assembleValues: row not owned by this rank");
        const LocalIndex lrow = static_cast<LocalIndex>(r - begin);
        const LocalIndex lcol = layout_.localIndex(coo_cols_[e]);
        check(lcol != kInvalidLocalIndex, layout_.comm(),
              "BsrMatrix::assembleValues: column outside the ghost pattern");
        const LocalIndex slot = detail::find_slot(row_ptr_, cols_, lrow, lcol);
        check(slot != kInvalidLocalIndex, layout_.comm(),
              "BsrMatrix::assembleValues: entry not present in the fixed structure");
        double* dst = values_.data() + static_cast<std::size_t>(slot) * vpe;
        const double* src = coo_vals_.data() + e * vpe;
        for (std::size_t v = 0; v < vpe; ++v) dst[v] += src[v];
    }
    coo_rows_.clear();
    coo_cols_.clear();
    coo_vals_.clear();
}

void BsrMatrix::apply(const Vector& x, Vector& y) const {
    check(assembled_, layout_.comm(), "BsrMatrix::apply: matrix not assembled");
    check(x.blockSize() == bs_ && y.blockSize() == bs_, layout_.comm(),
          "BsrMatrix::apply: vector block size mismatch");
    check(x.layout().compatibleWith(layout_) && y.layout().compatibleWith(layout_),
          layout_.comm(), "BsrMatrix::apply: incompatible vector layout");
    check(&x != &y, layout_.comm(), "BsrMatrix::apply: x and y must be distinct");

    detail::bsr_spmv_dispatch(layout_.localSize(), bs_, row_ptr_.data(), cols_.data(),
                              values_.data(), x.data(), y.data());
}

}  // namespace cfd::linalg

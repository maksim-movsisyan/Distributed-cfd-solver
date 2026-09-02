#include "cfd/linalg/csr_matrix.hpp"

#include <algorithm>

#include "assembly.hpp"
#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

CsrMatrix::CsrMatrix(MPI_Comm comm, GlobalIndex n_global_rows, LocalIndex n_local_rows)
    : layout_(comm, n_global_rows, n_local_rows) {}

void CsrMatrix::reserve(std::size_t nnz_hint) {
    coo_rows_.reserve(nnz_hint);
    coo_cols_.reserve(nnz_hint);
    coo_vals_.reserve(nnz_hint);
}

void CsrMatrix::addValue(GlobalIndex row, GlobalIndex col, double value) {
    // Staging is always allowed: assemble() consumes the entries into a new
    // structure, assembleValues() adds them into the existing one.
    check(row >= layout_.localBegin() && row < layout_.localBegin() + layout_.localSize(),
          layout_.comm(), "CsrMatrix::addValue: row not owned by this rank");
    check(col >= 0 && col < layout_.globalSize(), layout_.comm(),
          "CsrMatrix::addValue: column outside the global range");
    coo_rows_.push_back(row);
    coo_cols_.push_back(col);
    coo_vals_.push_back(value);
}

void CsrMatrix::assemble() {
    check(!assembled_, layout_.comm(), "CsrMatrix::assemble: already assembled");

    // Ghost set = foreign columns referenced by this rank.
    std::vector<GlobalIndex> ghosts;
    const GlobalIndex begin = layout_.localBegin();
    const GlobalIndex end = begin + layout_.localSize();
    for (GlobalIndex c : coo_cols_) {
        if (c < begin || c >= end) ghosts.push_back(c);
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    layout_.setGhosts(ghosts);  // collective

    detail::assemble_coo(layout_, 1, coo_rows_, coo_cols_, coo_vals_, row_ptr_, cols_, values_,
                         diag_idx_);

    coo_rows_.clear();
    coo_cols_.clear();
    coo_vals_.clear();
    coo_rows_.shrink_to_fit();
    coo_cols_.shrink_to_fit();
    coo_vals_.shrink_to_fit();
    assembled_ = true;
}

void CsrMatrix::assembleValues() {
    check(assembled_, layout_.comm(),
          "CsrMatrix::assembleValues: structure unknown, call assemble() first");
    const GlobalIndex begin = layout_.localBegin();
    const std::size_t n = coo_rows_.size();
    check(coo_vals_.size() == n, layout_.comm(),
          "CsrMatrix::assembleValues: malformed COO staging arrays");
    for (std::size_t e = 0; e < n; ++e) {
        const GlobalIndex r = coo_rows_[e];
        check(r >= begin && r < begin + layout_.localSize(), layout_.comm(),
              "CsrMatrix::assembleValues: row not owned by this rank");
        const LocalIndex lrow = static_cast<LocalIndex>(r - begin);
        const LocalIndex lcol = layout_.localIndex(coo_cols_[e]);
        check(lcol != kInvalidLocalIndex, layout_.comm(),
              "CsrMatrix::assembleValues: column outside the ghost pattern");
        const LocalIndex slot = detail::find_slot(row_ptr_, cols_, lrow, lcol);
        check(slot != kInvalidLocalIndex, layout_.comm(),
              "CsrMatrix::assembleValues: entry not present in the fixed structure");
        values_[static_cast<std::size_t>(slot)] += coo_vals_[e];
    }
    coo_rows_.clear();
    coo_cols_.clear();
    coo_vals_.clear();
}

void CsrMatrix::apply(const Vector& x, Vector& y) const {
    check(assembled_, layout_.comm(), "CsrMatrix::apply: matrix not assembled");
    check(x.blockSize() == 1 && y.blockSize() == 1, layout_.comm(),
          "CsrMatrix::apply: block size must be 1");
    check(x.layout().compatibleWith(layout_) && y.layout().compatibleWith(layout_),
          layout_.comm(), "CsrMatrix::apply: incompatible vector layout");
    check(&x != &y, layout_.comm(), "CsrMatrix::apply: x and y must be distinct");

    const LocalIndex n = layout_.localSize();
    const LocalIndex* CFD_RESTRICT rp = row_ptr_.data();
    const LocalIndex* CFD_RESTRICT cj = cols_.data();
    const double* CFD_RESTRICT av = values_.data();
    const double* CFD_RESTRICT xv = x.data();
    double* CFD_RESTRICT yv = y.data();

    for (LocalIndex i = 0; i < n; ++i) {
        const LocalIndex k1 = rp[i + 1];
        double s = 0.0;
        for (LocalIndex k = rp[i]; k < k1; ++k) s += av[k] * xv[cj[k]];
        yv[i] = s;
    }
}

}  // namespace cfd::linalg

#include "cfd/linalg/ldu_matrix.hpp"

#include "cfd/linalg/ldu_helpers.hpp"
#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

void LduMatrix::apply(const Vector& x, Vector& y, double alpha, double beta) const {
    check(assembled_, layout_.comm(), "LduMatrix::apply: matrix not assembled");
    check(x.blockSize() == 1 && y.blockSize() == 1, layout_.comm(),
          "LduMatrix::apply: vector block size mismatch (scalar operator expected)");
    check(x.layout().compatibleWith(layout_) && y.layout().compatibleWith(layout_),
          layout_.comm(), "LduMatrix::apply: incompatible vector layout");
    check(&x != &y, layout_.comm(), "LduMatrix::apply: x and y must be distinct");

    detail::ldu_spmv(layout_.localSize(), n_internal_edges_, owner_.size(),
                     owner_.data(), neigh_.data(), diag_.data(), upper_.data(),
                     lower_.data(), x.data(), y.data(), alpha, beta);
}

void LduMatrix::setZero() noexcept {
    std::fill(diag_.begin(), diag_.end(), 0.0);
    std::fill(upper_.begin(), upper_.end(), 0.0);
    std::fill(lower_.begin(), lower_.end(), 0.0);
}

void LduMatrix::assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                         const std::vector<LocalIndex>& dual_graph_off,
                         const std::vector<LocalIndex>& dual_graph_val) {
    check(!assembled_, layout_.comm(), "LduMatrix::assemble: matrix is already assembled");

    const std::size_t n_own = dual_graph_off.size() - 1;
    check(layout_.localSize() == static_cast<LocalIndex>(n_own), layout_.comm(),
          "LduMatrix::assemble: layout local size does not match graph row count");

    detail::ldu_assemble_from_dual_graph(dual_graph_off, dual_graph_val,
                                         owner_start_, owner_, neigh_,
                                         n_internal_edges_);

    // Allocate value arrays
    diag_.assign(n_own, 0.0);
    upper_.assign(owner_.size(), 0.0);
    lower_.assign(n_internal_edges_, 0.0);

    layout_.setGhosts(sorted_unique_ghost_gids);

    assembled_ = true;
}

void LduMatrix::assemble(const std::vector<GlobalIndex>& sorted_unique_ghost_gids,
                         const std::vector<LocalIndex>& face_owner,
                         const std::vector<LocalIndex>& face_neigh,
                         std::size_t n_internal_faces) {
    check(!assembled_, layout_.comm(), "LduMatrix::assemble: matrix is already assembled");

    const auto n_own = static_cast<std::size_t>(layout_.localSize());

    detail::ldu_assemble_from_faces(n_own, face_owner, face_neigh, n_internal_faces,
                                    owner_start_, owner_, neigh_,
                                    n_internal_edges_);

    diag_.assign(n_own, 0.0);
    upper_.assign(owner_.size(), 0.0);
    lower_.assign(n_internal_edges_, 0.0);

    layout_.setGhosts(sorted_unique_ghost_gids);

    assembled_ = true;
}

} // namespace cfd::linalg
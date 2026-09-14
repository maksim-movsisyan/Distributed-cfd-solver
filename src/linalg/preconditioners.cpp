#include "cfd/linalg/preconditioners.hpp"

#include <algorithm>
#include <sstream>

#include "cfd/linalg/bsr_helpers.hpp"
#include "cfd/linalg/bsr_matrix.hpp"
#include "cfd/linalg/csr_matrix.hpp"
#include "cfd/linalg/ldu_matrix.hpp"    
#include "cfd/linalg/ldu_helpers.hpp"
#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

// --- IdentityPreconditioner --------------------------------------------------

void IdentityPreconditioner::setup(const LinearOperator& op) { op_ = &op; }

void IdentityPreconditioner::apply(const Vector& r, Vector& z) const {
    check(op_ != nullptr, r.layout().comm(), "IdentityPreconditioner: setup() was not called");
    check(z.blockSize() == r.blockSize() && z.layout().compatibleWith(r.layout()),
          r.layout().comm(), "IdentityPreconditioner::apply: incompatible vectors");
    z.copyFrom(r);
    z.updateGhosts();  // contract: z leaves with current ghosts
}

// --- SgsPreconditioner -------------------------------------------------------

void SgsPreconditioner::setup(const LinearOperator& op) {
    csr_ = dynamic_cast<const CsrMatrix*>(&op);
    bsr_ = dynamic_cast<const BsrMatrix*>(&op);
    ldu_ = dynamic_cast<const LduMatrix*>(&op);
    
    if (csr_ != nullptr) {
        mode_ = Mode::Csr;
        check(csr_->assembled(), op.layout().comm(), "SgsPreconditioner: matrix not assembled");
        const LocalIndex n = csr_->localRows();
        inv_diag_.resize(static_cast<std::size_t>(n));
        const auto& diag = csr_->diagIndex();
        const auto& values = csr_->values();
        for (LocalIndex i = 0; i < n; ++i) {
            const LocalIndex d = diag[static_cast<std::size_t>(i)];
            if (d == kInvalidLocalIndex) {
                std::ostringstream os;
                os << "SgsPreconditioner: row " << op.layout().localBegin() + i
                   << " has no diagonal entry";
                fatal(op.layout().comm(), os.str());
            }
            const double v = values[static_cast<std::size_t>(d)];
            if (v == 0.0) {
                std::ostringstream os;
                os << "SgsPreconditioner: zero diagonal at row "
                   << op.layout().localBegin() + i;
                fatal(op.layout().comm(), os.str());
            }
            inv_diag_[static_cast<std::size_t>(i)] = 1.0 / v;
        }
        diag_lu_.clear();
        diag_pivots_.clear();
        return;
    }
    if (bsr_ != nullptr) {
        mode_ = Mode::Bsr;
        check(bsr_->assembled(), op.layout().comm(), "SgsPreconditioner: matrix not assembled");
        const LocalIndex n = bsr_->localRows();
        const int bs = bsr_->blockSize();
        const std::size_t bs2 = static_cast<std::size_t>(bs) * static_cast<std::size_t>(bs);
        diag_lu_.assign(static_cast<std::size_t>(n) * bs2, 0.0);
        diag_pivots_.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(bs), 0);
        const auto& diag = bsr_->diagIndex();
        const auto& values = bsr_->values();
        for (LocalIndex i = 0; i < n; ++i) {
            const LocalIndex d = diag[static_cast<std::size_t>(i)];
            if (d == kInvalidLocalIndex) {
                std::ostringstream os;
                os << "SgsPreconditioner: block row " << op.layout().localBegin() + i
                   << " has no diagonal block";
                fatal(op.layout().comm(), os.str());
            }
            double* lu = diag_lu_.data() + static_cast<std::size_t>(i) * bs2;
            const double* src = values.data() + static_cast<std::size_t>(d) * bs2;
            std::copy(src, src + bs2, lu);
            LocalIndex* piv = diag_pivots_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
            if (!detail::lu_factor_block(lu, bs, piv)) {
                std::ostringstream os;
                os << "SgsPreconditioner: singular diagonal block at row "
                   << op.layout().localBegin() + i;
                fatal(op.layout().comm(), os.str());
            }
        }
        inv_diag_.clear();
        return;
    }
    if (ldu_ != nullptr) {
        mode_ = Mode::Ldu;
        check(ldu_->assembled(), op.layout().comm(), "SgsPreconditioner: matrix not assembled");
        const LocalIndex n = ldu_->localRows();
        const auto n_sz = static_cast<std::size_t>(n);

        inv_diag_.resize(n_sz);
        work_y_.resize(n_sz);

        const auto& d = ldu_->diag();
        for (LocalIndex i = 0; i < n; ++i) {
            const double v = d[static_cast<std::size_t>(i)];
            if (v == 0.0) {
                std::ostringstream os;
                os << "SgsPreconditioner: zero diagonal at row " << op.layout().localBegin() + i;
                fatal(op.layout().comm(), os.str());
            }
            inv_diag_[static_cast<std::size_t>(i)] = 1.0 / v;
        }

        diag_lu_.clear();
        diag_pivots_.clear();
        return;
    }
    
    fatal(op.layout().comm(),
      "SgsPreconditioner::setup: operator is neither CsrMatrix, BsrMatrix, nor LduMatrix");
}

void SgsPreconditioner::apply(const Vector& r, Vector& z) const {
    const MPI_Comm comm = r.layout().comm();
    check(mode_ != Mode::None, comm, "SgsPreconditioner: setup() was not called");
    check(sweeps_ >= 1, comm, "SgsPreconditioner: sweeps must be >= 1");

    if (mode_ == Mode::Csr) {
        check(r.blockSize() == 1 && z.blockSize() == 1, comm,
              "SgsPreconditioner::apply: block size must be 1 for a CsrMatrix");
        check(r.layout().compatibleWith(csr_->layout()) && z.layout().compatibleWith(csr_->layout()),
              comm, "SgsPreconditioner::apply: incompatible vectors");

        const LocalIndex n = csr_->localRows();
        const LocalIndex* CFD_RESTRICT rp = csr_->rowPtr().data();
        const LocalIndex* CFD_RESTRICT cj = csr_->cols().data();
        const double* CFD_RESTRICT av = csr_->values().data();
        const LocalIndex* CFD_RESTRICT dg = csr_->diagIndex().data();
        const double* CFD_RESTRICT idg = inv_diag_.data();
        const double* CFD_RESTRICT rv = r.data();
        double* CFD_RESTRICT zv = z.data();

        // Start from z = 0 so that M^{-1} is one FIXED linear operator
        // (row i of a sweep reads rows ahead of the cursor; reusing the
        // previous apply's z would make the solver see a different
        // operator every iteration and diverge). Extra sweeps chain on the
        // result of the previous one.
        z.setZero();
        for (int sweep = 0; sweep < sweeps_; ++sweep) {
            for (LocalIndex i = 0; i < n; ++i) {
                const LocalIndex d = dg[i];
                double s = rv[i];
                for (LocalIndex k = rp[i]; k < rp[i + 1]; ++k) {
                    if (k != d) s -= av[k] * zv[cj[k]];
                }
                zv[i] = s * idg[i];
            }
            z.updateGhosts();
            for (LocalIndex i = n; i-- > 0;) {
                const LocalIndex d = dg[i];
                double s = rv[i];
                for (LocalIndex k = rp[i]; k < rp[i + 1]; ++k) {
                    if (k != d) s -= av[k] * zv[cj[k]];
                }
                zv[i] = s * idg[i];
            }
            z.updateGhosts();  // contract: z leaves with current ghosts
        }
        return;
    }
    if (mode_ == Mode::Bsr) {
        // BSR mode
        check(r.blockSize() == bsr_->blockSize() && z.blockSize() == bsr_->blockSize(), comm,
            "SgsPreconditioner::apply: vector block size mismatch");
        check(r.layout().compatibleWith(bsr_->layout()) && z.layout().compatibleWith(bsr_->layout()),
            comm, "SgsPreconditioner::apply: incompatible vectors");

        // Start from z = 0: fixed linear operator (see the CSR branch); extra
        // sweeps chain on the result of the previous one.
        z.setZero();
        for (int sweep = 0; sweep < sweeps_; ++sweep) {
            detail::bsr_sgs_sweep_dispatch(bsr_->localRows(), true, bsr_->blockSize(),
                                        bsr_->rowPtr().data(), bsr_->cols().data(),
                                        bsr_->values().data(), bsr_->diagIndex().data(),
                                        diag_lu_.data(), diag_pivots_.data(), r.data(), z.data());
            z.updateGhosts();
            detail::bsr_sgs_sweep_dispatch(bsr_->localRows(), false, bsr_->blockSize(),
                                        bsr_->rowPtr().data(), bsr_->cols().data(),
                                        bsr_->values().data(), bsr_->diagIndex().data(),
                                        diag_lu_.data(), diag_pivots_.data(), r.data(), z.data());
            z.updateGhosts();
        }
        return;
    }
    if (mode_ == Mode::Ldu) {
        check(r.blockSize() == 1 && z.blockSize() == 1, comm,
              "SgsPreconditioner::apply: block size must be 1 for LduMatrix");
        check(r.layout().compatibleWith(ldu_->layout()) && z.layout().compatibleWith(ldu_->layout()),
              comm, "SgsPreconditioner::apply: incompatible vectors");

        const LocalIndex n = ldu_->localRows();
        const std::size_t n_int = ldu_->numInternalEdges();
        const std::size_t n_total = ldu_->numTotalEdges();

        const LocalIndex* CFD_RESTRICT owner_start = ldu_->ownerStart().data();
        const LocalIndex* CFD_RESTRICT owner       = ldu_->owner().data();
        const LocalIndex* CFD_RESTRICT neigh       = ldu_->neigh().data();
        const double* CFD_RESTRICT idg             = inv_diag_.data();
        const double* CFD_RESTRICT up              = ldu_->upperData();
        const double* CFD_RESTRICT lo              = ldu_->lowerData();

        const double* CFD_RESTRICT rv = r.data();
        double* CFD_RESTRICT zv       = z.data();
        double* CFD_RESTRICT yv       = work_y_.data();

        z.setZero();

        for (int sweep = 0; sweep < sweeps_; ++sweep) {
            const bool first = (sweep == 0);

            // 1. Forward pass (D + L)
            detail::ldu_sgs_forward(n, n_int, n_total, owner_start, owner, neigh,
                                    idg, up, lo, rv, zv, yv, first);
            z.updateGhosts();

            // 2. Backward pass (D + U)
            detail::ldu_sgs_backward(n, n_int, n_total, owner_start, owner, neigh,
                                     idg, up, lo, rv, zv, yv);
            z.updateGhosts();
        }
        return;
    }
}

}  // namespace cfd::linalg

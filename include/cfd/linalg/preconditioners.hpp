#pragma once

#include "cfd/linalg/preconditioner.hpp"

namespace cfd::linalg {

class CsrMatrix;
class BsrMatrix;

/**
 * @class IdentityPreconditioner
 * @brief M = I: the null preconditioner (useful for testing and for measuring
 * preconditioner benefit). apply() is a copy plus one halo exchange to honor
 * the Preconditioner contract.
 */
class IdentityPreconditioner : public Preconditioner {
public:
    void setup(const LinearOperator& op) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    const LinearOperator* op_ = nullptr;
};

/**
 * @class SgsPreconditioner
 * @brief Symmetric Gauss–Seidel for CsrMatrix and BsrMatrix.
 *
 * Distributed variant (hybrid SGS) — the standard communication-cheap
 * approximation: each rank sweeps its owned rows forward, refreshes the halo,
 * sweeps backward, refreshes again. The update ordering across ranks differs
 * from the exact sequential SGS, but the preconditioner quality is close and
 * the cost is only two halo rounds per apply.
 *
 * Diagonals are inverted (CSR) / LU-factorized with partial pivoting (BSR) at
 * setup; rows with a structurally missing diagonal or a singular diagonal
 * block abort. setup() must be re-called after value reassembly.
 */
class SgsPreconditioner : public Preconditioner {
public:
    void setup(const LinearOperator& op) override;
    void apply(const Vector& r, Vector& z) const override;

    /// Extra forward+backward passes per apply (default 1; each costs two
    /// halo rounds and typically buys little).
    void setSweeps(int sweeps) { sweeps_ = sweeps; }

private:
    enum class Mode { None, Csr, Bsr };

    Mode mode_ = Mode::None;
    const CsrMatrix* csr_ = nullptr;
    const BsrMatrix* bsr_ = nullptr;
    int sweeps_ = 1;

    std::vector<double> inv_diag_;             // CSR: 1 / a_ii
    std::vector<double> diag_lu_;              // BSR: factored diagonal blocks
    std::vector<LocalIndex> diag_pivots_;      // BSR: pivots per block row
};

}  // namespace cfd::linalg

#include "cfd/linalg/bicgstab.hpp"

#include <cmath>
#include <cstdio>

#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

namespace {

const char* status_name(SolverStatus s) {
    switch (s) {
        case SolverStatus::Converged: return "converged";
        case SolverStatus::Diverged: return "diverged";
        case SolverStatus::MaxIterations: return "max-iterations";
    }
    return "?";
}

// Re-evaluate the monitored norm from the actual recurrence vector this often,
// so the reconstructed (fused) norm cannot drift unnoticed.
constexpr int kNormRefreshInterval = 32;

}  // namespace

void BiCGSTAB::ensureWorkspace(const Vector& prototype) {
    const bool ok = r_.blockSize() == prototype.blockSize() &&
                    r_.scalarSize() == prototype.scalarSize() &&
                    r_.layout().compatibleWith(prototype.layout());
    if (ok) return;
    r_ = Vector(prototype.layout(), prototype.blockSize());
    r0_ = r_;
    p_ = r_;
    v_ = r_;
    s_ = r_;
    t_ = r_;
    phat_ = r_;
    shat_ = r_;
}

IterationResult BiCGSTAB::solve(const LinearOperator& A, const Preconditioner& M, Vector& x,
                                const Vector& b) {
    const MPI_Comm comm = b.layout().comm();
    check(x.blockSize() == b.blockSize() && x.blockSize() == A.blockSize(), comm,
          "BiCGSTAB::solve: vector block sizes do not match the operator");
    check(x.layout().compatibleWith(b.layout()) && x.layout().compatibleWith(A.layout()), comm,
          "BiCGSTAB::solve: incompatible vector layout");
    check(x.localRows() == A.localRows(), comm, "BiCGSTAB::solve: row count mismatch");
    ensureWorkspace(x);

    last_ = IterationResult{};
    IterationResult& res = last_;
    const int max_it = params_.max_iterations;

    // r = b - A x (the initial guess may carry stale ghosts).
    x.updateGhosts();
    A.apply(x, v_);
    r_.copyFrom(b);
    r_.axpy(-1.0, v_);

    const double bnorm = b.norm2();
    double rnorm = r_.norm2();
    const double scale = (bnorm > 0.0) ? bnorm : 1.0;
    const double tol = std::max(params_.absolute_tolerance, params_.relative_tolerance * scale);
    res.initial_residual = rnorm;
    res.residual_history.reserve(static_cast<std::size_t>(max_it) + 1);

    const bool print_summary = params_.verbosity >= Verbosity::Summary && b.layout().rank() == 0;
    const bool print_iters = params_.verbosity >= Verbosity::Verbose && b.layout().rank() == 0;

    if (!std::isfinite(rnorm)) {
        res.status = SolverStatus::Diverged;
    } else {
        res.residual_history.push_back(rnorm / scale);
        if (rnorm <= tol) res.status = SolverStatus::Converged;
    }

    int it = 0;
    if (res.status == SolverStatus::MaxIterations) {
    r0_.copyFrom(r_);  // shadow residual: fixed, defines the bi-orthogonality
    v_.setZero();

    // rho = r0 . r == ||r||^2 here because r0 == r on entry.
    double rho = rnorm * rnorm;
    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    while (true) {
        if (it >= max_it) {
            res.status = SolverStatus::MaxIterations;
            break;
        }
        ++it;

        // Search direction: p = r + beta (p - omega v).
        if (it == 1) {
            p_.copyFrom(r_);
        } else {
            const double beta = (rho / rho_old) * (alpha / omega);
            if (!std::isfinite(beta)) {
                res.status = SolverStatus::Diverged;
                break;
            }
            p_.scale(beta);
            p_.axpy(-beta * omega, v_);
            p_.axpy(1.0, r_);
        }

        M.apply(p_, phat_);  // leaves phat ghosts current
        A.apply(phat_, v_);

        const double gamma = r0_.dot(v_);
        if (!std::isfinite(gamma) || gamma == 0.0) {
            res.status = SolverStatus::Diverged;  // Krylov breakdown
            break;
        }
        alpha = rho / gamma;

        // s = r - alpha v: the residual after the alpha-part of the update.
        s_.copyFrom(r_);
        s_.axpy(-alpha, v_);

        M.apply(s_, shat_);
        A.apply(shat_, t_);

        // Five fused inner products, one Allreduce:
        //   omega = (t.s) / (t.t), rho_next = r0.(s - omega t),
        //   ||r_next||^2 = ss - 2 omega ts + omega^2 tt.
        const std::pair<const Vector*, const Vector*> products[] = {
            {&t_, &t_}, {&t_, &s_}, {&r0_, &t_}, {&r0_, &s_}, {&s_, &s_}};
        double out[5];
        Vector::batchedDots(products, out);
        const double tt = out[0], ts = out[1], r0t = out[2], r0s = out[3], ss = out[4];

        if (!std::isfinite(ts) || !std::isfinite(tt) || tt == 0.0) {
            res.status = SolverStatus::Diverged;
            break;
        }
        const double omega_new = ts / tt;

        // x += alpha phat + omega shat;  r = s - omega t.
        x.axpy(alpha, phat_);
        x.axpy(omega_new, shat_);
        r_.copyFrom(s_);
        r_.axpy(-omega_new, t_);

        const double rn2 = ss - 2.0 * omega_new * ts + omega_new * omega_new * tt;
        rnorm = std::sqrt(std::max(rn2, 0.0));
        const double rho_new = r0s - omega_new * r0t;
        res.residual_history.push_back(rnorm / scale);
        if (print_iters) {
            std::printf("  [%s] it %4d  |r|/|b| = %.3e\n", name(), it, rnorm / scale);
        }

        if (!std::isfinite(rnorm) || !std::isfinite(omega_new) || omega_new == 0.0) {
            res.status = SolverStatus::Diverged;
            break;
        }
        if (it % kNormRefreshInterval == 0) {
            rnorm = r_.norm2();
            res.residual_history.back() = rnorm / scale;
        }
        if (rnorm <= tol) {
            res.status = SolverStatus::Converged;
            break;
        }
        if (!std::isfinite(rho_new) || rho_new == 0.0) {
            res.status = SolverStatus::Diverged;  // r orthogonal to the shadow
            break;
        }

        rho_old = rho;
        rho = rho_new;
        omega = omega_new;
    }
    }  // Krylov loop (skipped entirely when converged at iteration 0)

    res.iterations = it;
    res.final_residual = rnorm;

    if (params_.verify_final_residual) {
        x.updateGhosts();
        A.apply(x, v_);
        s_.copyFrom(b);
        s_.axpy(-1.0, v_);
        res.true_final_residual = s_.norm2();
    }

    if (print_summary) {
        std::printf("  [%s] %-14s iters %4d  |r|/|b| = %.3e", name(), status_name(res.status), it,
                    rnorm / scale);
        if (params_.verify_final_residual) {
            std::printf("  (true %.3e)", res.true_final_residual / scale);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    return res;
}

}  // namespace cfd::linalg

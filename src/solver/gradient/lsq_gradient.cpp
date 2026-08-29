#include "cfd/solver/gradient/lsq_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::solver::gradient {

namespace {

// Relative diagonal regularization for quasi-2D and degenerate stencils
constexpr double kRegScale = 1.0e-12;
constexpr double kAbsReg   = 1.0e-20;

/**
 * @brief Inverts symmetric 3x3 normal equation matrix via adjugate with regularization.
 */
void invert3x3(double a[9]) noexcept {
    const double max_diag = std::max({std::abs(a[0]), std::abs(a[4]), std::abs(a[8])});
    const double reg = kRegScale * max_diag + kAbsReg;

    a[0] += reg;
    a[4] += reg;
    a[8] += reg;

    const double c00 = a[4] * a[8] - a[5] * a[7];
    const double c01 = a[5] * a[6] - a[3] * a[8];
    const double c02 = a[3] * a[7] - a[4] * a[6];
    const double det = a[0] * c00 + a[1] * c01 + a[2] * c02;

    if (std::abs(det) < 1.0e-30) {
        std::fill(a, a + 9, 0.0);
        return;
    }

    const double inv_det = 1.0 / det;

    const double m[9] = {
        c00 * inv_det,
        (a[2] * a[7] - a[1] * a[8]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        c01 * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        (a[2] * a[3] - a[0] * a[5]) * inv_det,
        c02 * inv_det,
        (a[1] * a[6] - a[0] * a[7]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det
    };

    for (int i = 0; i < 9; ++i) {
        a[i] = m[i];
    }
}

} // anonymous namespace

LsqGradient::LsqGradient(const mesh::MeshPart& mp, const VertexAdjacency& adj)
    : adj_(adj),
      n_own_(static_cast<std::size_t>(mp.n_own)),
      mat9_(9 * static_cast<std::size_t>(mp.n_own), 0.0) {
    const auto total_entries = adj_.dx.size();
    wdx_.resize(total_entries);
    wdy_.resize(total_entries);
    wdz_.resize(total_entries);

    // 1. Precompute pre-weighted vectors: w_j * dx_j
    for (std::size_t j = 0; j < total_entries; ++j) {
        const double w = adj_.w[j];
        wdx_[j] = w * adj_.dx[j];
        wdy_[j] = w * adj_.dy[j];
        wdz_[j] = w * adj_.dz[j];
    }

    // 2. Assemble and invert normal matrices for each owned cell
    for (std::size_t c = 0; c < n_own_; ++c) {
        double a[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        for (LocalIndex j = adj_.offsets[c]; j < adj_.offsets[c + 1]; ++j) {
            const auto js = static_cast<std::size_t>(j);
            const double wx = wdx_[js];
            const double wy = wdy_[js];
            const double wz = wdz_[js];
            const double ex = adj_.dx[js];
            const double ey = adj_.dy[js];
            const double ez = adj_.dz[js];

            a[0] += wx * ex;
            a[1] += wx * ey;
            a[2] += wx * ez;
            a[3] += wy * ex;
            a[4] += wy * ey;
            a[5] += wy * ez;
            a[6] += wz * ex;
            a[7] += wz * ey;
            a[8] += wz * ez;
        }

        invert3x3(a);
        std::copy(a, a + 9, mat9_.begin() + static_cast<std::ptrdiff_t>(9 * c));
    }
}

void LsqGradient::compute(fields::ConstPrimitiveView q,
                         fields::PrimitiveGradView<double> grad) const {
    const std::size_t stride = grad.stride;

    const LocalIndex* CFD_RESTRICT nbo = adj_.offsets.data();
    const LocalIndex* CFD_RESTRICT nbc = adj_.cells.data();

    const double* CFD_RESTRICT wdx = wdx_.data();
    const double* CFD_RESTRICT wdy = wdy_.data();
    const double* CFD_RESTRICT wdz = wdz_.data();
    const double* CFD_RESTRICT mat = mat9_.data();

    const double* CFD_RESTRICT prs = q.prs;
    const double* CFD_RESTRICT vx  = q.vx;
    const double* CFD_RESTRICT vy  = q.vy;
    const double* CFD_RESTRICT vz  = q.vz;
    const double* CFD_RESTRICT tmp = q.tmp;

    for (std::size_t c = 0; c < n_own_; ++c) {
        // rhs[dim][var]: dimension-major 3x5 accumulator kept entirely in registers
        double r0_p = 0.0, r0_u = 0.0, r0_v = 0.0, r0_w = 0.0, r0_t = 0.0;
        double r1_p = 0.0, r1_u = 0.0, r1_v = 0.0, r1_w = 0.0, r1_t = 0.0;
        double r2_p = 0.0, r2_u = 0.0, r2_v = 0.0, r2_w = 0.0, r2_t = 0.0;

        const double p0 = prs[c];
        const double u0 = vx[c];
        const double v0 = vy[c];
        const double w0 = vz[c];
        const double t0 = tmp[c];

        const LocalIndex j_beg = nbo[c];
        const LocalIndex j_end = nbo[c + 1];

        for (LocalIndex j = j_beg; j < j_end; ++j) {
            const auto js = static_cast<std::size_t>(j);
            const auto cs = static_cast<std::size_t>(nbc[js]);

            const double dp = prs[cs] - p0;
            const double du = vx[cs]  - u0;
            const double dv = vy[cs]  - v0;
            const double dw = vz[cs]  - w0;
            const double dt = tmp[cs] - t0;

            const double wx = wdx[js];
            const double wy = wdy[js];
            const double wz = wdz[js];

            // Accumulate X-derivatives RHS
            r0_p += wx * dp;
            r0_u += wx * du;
            r0_v += wx * dv;
            r0_w += wx * dw;
            r0_t += wx * dt;

            // Accumulate Y-derivatives RHS
            r1_p += wy * dp;
            r1_u += wy * du;
            r1_v += wy * dv;
            r1_w += wy * dw;
            r1_t += wy * dt;

            // Accumulate Z-derivatives RHS
            r2_p += wz * dp;
            r2_u += wz * du;
            r2_v += wz * dv;
            r2_w += wz * dw;
            r2_t += wz * dt;
        }

        // Multiply: grad = A^{-1} * rhs
        const double* CFD_RESTRICT m = mat + 9 * c;

        const double m00 = m[0], m01 = m[1], m02 = m[2];
        const double m10 = m[3], m11 = m[4], m12 = m[5];
        const double m20 = m[6], m21 = m[7], m22 = m[8];

        // Pressure gradient (dx, dy, dz)
        grad.prs_grad[c]              = m00 * r0_p + m01 * r1_p + m02 * r2_p;
        grad.prs_grad[stride + c]     = m10 * r0_p + m11 * r1_p + m12 * r2_p;
        grad.prs_grad[2 * stride + c] = m20 * r0_p + m21 * r1_p + m22 * r2_p;

        // Velocity-X gradient
        grad.vx_grad[c]               = m00 * r0_u + m01 * r1_u + m02 * r2_u;
        grad.vx_grad[stride + c]      = m10 * r0_u + m11 * r1_u + m12 * r2_u;
        grad.vx_grad[2 * stride + c]  = m20 * r0_u + m21 * r1_u + m22 * r2_u;

        // Velocity-Y gradient
        grad.vy_grad[c]               = m00 * r0_v + m01 * r1_v + m02 * r2_v;
        grad.vy_grad[stride + c]      = m10 * r0_v + m11 * r1_v + m12 * r2_v;
        grad.vy_grad[2 * stride + c]  = m20 * r0_v + m21 * r1_v + m22 * r2_v;

        // Velocity-Z gradient
        grad.vz_grad[c]               = m00 * r0_w + m01 * r1_w + m02 * r2_w;
        grad.vz_grad[stride + c]      = m10 * r0_w + m11 * r1_w + m12 * r2_w;
        grad.vz_grad[2 * stride + c]  = m20 * r0_w + m21 * r1_w + m22 * r2_w;

        // Temperature gradient
        grad.tmp_grad[c]              = m00 * r0_t + m01 * r1_t + m02 * r2_t;
        grad.tmp_grad[stride + c]     = m10 * r0_t + m11 * r1_t + m12 * r2_t;
        grad.tmp_grad[2 * stride + c] = m20 * r0_t + m21 * r1_t + m22 * r2_t;
    }
}

} // namespace cfd::solver::gradient
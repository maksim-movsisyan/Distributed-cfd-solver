// BSR hot kernels: block SpMV and block Gauss-Seidel sweeps, plus small dense
// block LU. Template instantiations for block sizes 1..8 give fully unrolled,
// SIMD-friendly inner loops (the CFD sweet spot, e.g. 5 conservation vars);
// larger blocks take the dynamic fallback. Raw pointers only: these functions
// sit on the innermost loop of the solver.
#pragma once

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"

namespace cfd::linalg::detail {

// --- small dense block LU (setup time) ---------------------------------------

/// In-place LU with partial pivoting of a row-major bs x bs block.
/// Returns false for a (numerically) singular block.
inline bool lu_factor_block(double* a, int bs, LocalIndex* pivots) {
    for (int k = 0; k < bs; ++k) {
        int m = k;
        double mx = std::fabs(a[static_cast<std::size_t>(k * bs) + static_cast<std::size_t>(k)]);
        for (int i = k + 1; i < bs; ++i) {
            const double v = std::fabs(a[static_cast<std::size_t>(i * bs) + static_cast<std::size_t>(k)]);
            if (v > mx) {
                mx = v;
                m = i;
            }
        }
        pivots[k] = m;
        if (mx == 0.0) return false;
        if (m != k) {
            for (int j = 0; j < bs; ++j) {
                std::swap(a[static_cast<std::size_t>(k * bs) + static_cast<std::size_t>(j)],
                          a[static_cast<std::size_t>(m * bs) + static_cast<std::size_t>(j)]);
            }
        }
        const double d = a[static_cast<std::size_t>(k * bs) + static_cast<std::size_t>(k)];
        for (int i = k + 1; i < bs; ++i) {
            const double l = a[static_cast<std::size_t>(i * bs) + static_cast<std::size_t>(k)] / d;
            a[static_cast<std::size_t>(i * bs) + static_cast<std::size_t>(k)] = l;
            if (l != 0.0) {
                for (int j = k + 1; j < bs; ++j) {
                    a[static_cast<std::size_t>(i * bs) + static_cast<std::size_t>(j)] -=
                        l * a[static_cast<std::size_t>(k * bs) + static_cast<std::size_t>(j)];
                }
            }
        }
    }
    return true;
}

/// Solves (L U) b = rhs in place, compile-time block size BS.
template <int BS>
inline void lu_solve_block(const double* CFD_RESTRICT lu, const LocalIndex* CFD_RESTRICT piv,
                           double* CFD_RESTRICT b) {
    if constexpr (BS > 1) {
        // For BS == 1 the pivot is identity by construction.
        for (int k = 0; k < BS; ++k) {
            if (piv[k] != k) std::swap(b[k], b[piv[k]]);
        }
    }
    for (int i = 1; i < BS; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= lu[i * BS + j] * b[j];
        b[i] = s;
    }
    for (int i = BS - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < BS; ++j) s -= lu[i * BS + j] * b[j];
        b[i] = s / lu[i * BS + i];
    }
}

/// Same for a runtime block size.
inline void lu_solve_block_dyn(int bs, const double* CFD_RESTRICT lu,
                               const LocalIndex* CFD_RESTRICT piv, double* CFD_RESTRICT b) {
    for (int k = 0; k < bs; ++k) {
        if (piv[k] != k) std::swap(b[k], b[piv[k]]);
    }
    for (int i = 1; i < bs; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) {
            s -= lu[static_cast<std::size_t>(i) * static_cast<std::size_t>(bs) + static_cast<std::size_t>(j)] * b[j];
        }
        b[i] = s;
    }
    for (int i = bs - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < bs; ++j) {
            s -= lu[static_cast<std::size_t>(i) * static_cast<std::size_t>(bs) + static_cast<std::size_t>(j)] * b[j];
        }
        b[i] = s / lu[static_cast<std::size_t>(i) * static_cast<std::size_t>(bs) + static_cast<std::size_t>(i)];
    }
}

// --- block SpMV --------------------------------------------------------------

template <int BS>
inline void bsr_spmv(LocalIndex n_rows, const LocalIndex* CFD_RESTRICT row_ptr,
                     const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                     const double* CFD_RESTRICT x, double* CFD_RESTRICT y) {
    constexpr int BS2 = BS * BS;
    for (LocalIndex i = 0; i < n_rows; ++i) {
        const LocalIndex k1 = row_ptr[i + 1];
        double acc[BS];
        for (int ii = 0; ii < BS; ++ii) acc[ii] = 0.0;
        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            const double* CFD_RESTRICT a =
                values + static_cast<std::size_t>(k) * static_cast<std::size_t>(BS2);
            const double* CFD_RESTRICT xb =
                x + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) {
                double s = 0.0;
                for (int jj = 0; jj < BS; ++jj) s += a[ii * BS + jj] * xb[jj];
                acc[ii] += s;
            }
        }
        double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);
        for (int ii = 0; ii < BS; ++ii) yb[ii] = acc[ii];
    }
}

inline void bsr_spmv_dyn(LocalIndex n_rows, int bs, const LocalIndex* CFD_RESTRICT row_ptr,
                         const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                         const double* CFD_RESTRICT x, double* CFD_RESTRICT y) {
    const std::size_t bs2 = static_cast<std::size_t>(bs) * static_cast<std::size_t>(bs);
    for (LocalIndex i = 0; i < n_rows; ++i) {
        const LocalIndex k1 = row_ptr[i + 1];
        double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
        for (int ii = 0; ii < bs; ++ii) yb[ii] = 0.0;
        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            const double* CFD_RESTRICT a =
                values + static_cast<std::size_t>(k) * bs2;
            const double* CFD_RESTRICT xb =
                x + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(bs);
            for (int ii = 0; ii < bs; ++ii) {
                double s = 0.0;
                for (int jj = 0; jj < bs; ++jj) {
                    s += a[static_cast<std::size_t>(ii) * static_cast<std::size_t>(bs) + static_cast<std::size_t>(jj)] * xb[static_cast<std::size_t>(jj)];
                }
                yb[ii] += s;
            }
        }
    }
}

inline void bsr_spmv_dispatch(LocalIndex n_rows, int bs, const LocalIndex* CFD_RESTRICT row_ptr,
                              const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                              const double* CFD_RESTRICT x, double* CFD_RESTRICT y) {
    switch (bs) {
        case 1: bsr_spmv<1>(n_rows, row_ptr, cols, values, x, y); break;
        case 2: bsr_spmv<2>(n_rows, row_ptr, cols, values, x, y); break;
        case 3: bsr_spmv<3>(n_rows, row_ptr, cols, values, x, y); break;
        case 4: bsr_spmv<4>(n_rows, row_ptr, cols, values, x, y); break;
        case 5: bsr_spmv<5>(n_rows, row_ptr, cols, values, x, y); break;
        case 6: bsr_spmv<6>(n_rows, row_ptr, cols, values, x, y); break;
        case 7: bsr_spmv<7>(n_rows, row_ptr, cols, values, x, y); break;
        case 8: bsr_spmv<8>(n_rows, row_ptr, cols, values, x, y); break;
        default: bsr_spmv_dyn(n_rows, bs, row_ptr, cols, values, x, y); break;
    }
}

// --- block symmetric Gauss-Seidel sweeps -------------------------------------
//
// One row update: z_i = D_i^{-1} (r_i - sum_{k != diag} A_ik z_k), where D_i is
// the LU-factored diagonal block. Ghost slots of z (and, in the backward
// sweep, rows already touched by the forward sweep) carry their latest values.

template <int BS>
inline void bsr_sgs_sweep(LocalIndex n_rows, bool forward, const LocalIndex* CFD_RESTRICT row_ptr,
                          const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                          const LocalIndex* CFD_RESTRICT diag_idx, const double* CFD_RESTRICT diag_lu,
                          const LocalIndex* CFD_RESTRICT diag_piv, const double* CFD_RESTRICT r,
                          double* CFD_RESTRICT z) {
    constexpr int BS2 = BS * BS;
    for (LocalIndex cnt = 0; cnt < n_rows; ++cnt) {
        const LocalIndex i = forward ? cnt : static_cast<LocalIndex>(n_rows - 1 - cnt);
        const LocalIndex k1 = row_ptr[i + 1];
        const LocalIndex d = diag_idx[i];
        const double* CFD_RESTRICT rb = r + static_cast<std::size_t>(i) * BS;
        double s[BS];
        for (int ii = 0; ii < BS; ++ii) s[ii] = rb[ii];
        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            if (k == d) continue;
            const double* CFD_RESTRICT a =
                values + static_cast<std::size_t>(k) * static_cast<std::size_t>(BS2);
            const double* CFD_RESTRICT zb =
                z + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) {
                double t = 0.0;
                for (int jj = 0; jj < BS; ++jj) t += a[ii * BS + jj] * zb[jj];
                s[ii] -= t;
            }
        }
        lu_solve_block<BS>(diag_lu + static_cast<std::size_t>(i) * BS2,
                           diag_piv + static_cast<std::size_t>(i) * BS, s);
        double* CFD_RESTRICT zi = z + static_cast<std::size_t>(i) * BS;
        for (int ii = 0; ii < BS; ++ii) zi[ii] = s[ii];
    }
}

inline void bsr_sgs_sweep_dyn(LocalIndex n_rows, bool forward, int bs,
                              const LocalIndex* CFD_RESTRICT row_ptr, const LocalIndex* CFD_RESTRICT cols,
                              const double* CFD_RESTRICT values, const LocalIndex* CFD_RESTRICT diag_idx,
                              const double* CFD_RESTRICT diag_lu, const LocalIndex* CFD_RESTRICT diag_piv,
                              const double* CFD_RESTRICT r, double* CFD_RESTRICT z) {
    const std::size_t bs2 = static_cast<std::size_t>(bs) * static_cast<std::size_t>(bs);
    static thread_local std::vector<double> s_buf;
    s_buf.resize(static_cast<std::size_t>(bs));
    double* s = s_buf.data();
    for (LocalIndex cnt = 0; cnt < n_rows; ++cnt) {
        const LocalIndex i = forward ? cnt : static_cast<LocalIndex>(n_rows - 1 - cnt);
        const LocalIndex k1 = row_ptr[i + 1];
        const LocalIndex d = diag_idx[i];
        const double* CFD_RESTRICT rb = r + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
        for (int ii = 0; ii < bs; ++ii) s[ii] = rb[ii];
        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            if (k == d) continue;
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * bs2;
            const double* CFD_RESTRICT zb =
                z + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(bs);
            for (int ii = 0; ii < bs; ++ii) {
                double t = 0.0;
                for (int jj = 0; jj < bs; ++jj) {
                    t += a[static_cast<std::size_t>(ii) * static_cast<std::size_t>(bs) + static_cast<std::size_t>(jj)] * zb[jj];
                }
                s[ii] -= t;
            }
        }
        lu_solve_block_dyn(bs, diag_lu + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs2),
                           diag_piv + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs), s);
        double* CFD_RESTRICT zi = z + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
        for (int ii = 0; ii < bs; ++ii) zi[ii] = s[ii];
    }
}

inline void bsr_sgs_sweep_dispatch(LocalIndex n_rows, bool forward, int bs,
                                   const LocalIndex* CFD_RESTRICT row_ptr,
                                   const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                                   const LocalIndex* CFD_RESTRICT diag_idx, const double* CFD_RESTRICT diag_lu,
                                   const LocalIndex* CFD_RESTRICT diag_piv, const double* CFD_RESTRICT r,
                                   double* CFD_RESTRICT z) {
    switch (bs) {
        case 1: bsr_sgs_sweep<1>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 2: bsr_sgs_sweep<2>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 3: bsr_sgs_sweep<3>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 4: bsr_sgs_sweep<4>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 5: bsr_sgs_sweep<5>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 6: bsr_sgs_sweep<6>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 7: bsr_sgs_sweep<7>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        case 8: bsr_sgs_sweep<8>(n_rows, forward, row_ptr, cols, values, diag_idx, diag_lu, diag_piv, r, z); break;
        default:
            bsr_sgs_sweep_dyn(n_rows, forward, bs, row_ptr, cols, values, diag_idx, diag_lu,
                              diag_piv, r, z);
            break;
    }
}

}  // namespace cfd::linalg::detail

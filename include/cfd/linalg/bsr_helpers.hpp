// COO -> local CSR assembly shared by CsrMatrix and BsrMatrix.
// `values_per_edge` doubles live on each graph edge: 1 for CSR, bs*bs for BSR.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <cassert>


#include "cfd/core/types.hpp"

namespace cfd::linalg::detail {

// bsr matrix initialization from ajecency dual graph
inline void bsr_assemble(const std::vector<LocalIndex>& dual_graph_off,
                         const std::vector<LocalIndex>& dual_graph_val,
                         std::vector<LocalIndex>& row_ptr_,
                         std::vector<LocalIndex>& diag_idx_,
                         std::vector<LocalIndex>& cols_) {
    // 0. get number of graph vertices (matrix rows == owned cells)
    const std::size_t n_own = dual_graph_off.size() - 1;

    row_ptr_.resize(n_own + 1, 0);
    diag_idx_.resize(n_own);
    
    // 1. counting number of cols per each row (== 1 + number of neighbors)
    for (std::size_t c = 0; c < n_own; ++c) { 
        const LocalIndex ncols = 1 + (dual_graph_off[c + 1] - dual_graph_off[c]);
        row_ptr_[c + 1] = row_ptr_[c] + ncols;
    }

    // 2. fill column indices
    const std::size_t nnz = static_cast<std::size_t>(row_ptr_[n_own]);
    cols_.resize(nnz);
    

    // loop over all own graph vertices
    for (std::size_t c = 0; c < n_own; ++c) {
        const std::size_t row_start_pos = static_cast<std::size_t>(row_ptr_[c]);
        std::size_t write_pos = row_start_pos;

        cols_[write_pos++] = static_cast<LocalIndex>(c);

        const std::size_t ptr1 = static_cast<std::size_t>(dual_graph_off[c]);
        const std::size_t ptr2 = static_cast<std::size_t>(dual_graph_off[c + 1]);

        for (std::size_t e = ptr1; e < ptr2; ++e) {
            cols_[write_pos++] = dual_graph_val[e];
        }

        auto it_beg = cols_.begin() + static_cast<std::ptrdiff_t>(row_start_pos);
        auto it_end = cols_.begin() + static_cast<std::ptrdiff_t>(write_pos);
        std::sort(it_beg, it_end);

        auto it = std::lower_bound(it_beg, it_end, static_cast<LocalIndex>(c));
        diag_idx_[c] = static_cast<LocalIndex>(row_start_pos) + static_cast<LocalIndex>(std::distance(it_beg, it));
    }
}





// cols are supposed to be sorted
inline std::size_t bsr_get_pos(const LocalIndex* CFD_RESTRICT row_ptr_,
                               const LocalIndex* CFD_RESTRICT cols_,
                               LocalIndex row, LocalIndex col) {
    const std::size_t ptr1 = static_cast<std::size_t>(row_ptr_[row]);
    const std::size_t ptr2 = static_cast<std::size_t>(row_ptr_[row + 1]);     
    
    const LocalIndex* row_start = cols_ + ptr1;
    const LocalIndex* row_end   = cols_ + ptr2;

    const LocalIndex* it = std::lower_bound(row_start, row_end, col);
 
    assert(it != row_end && *it == col && "CFD Error: Element missing in dual graph!");

    return static_cast<std::size_t>(it - cols_);
}





// bsr general SpMV y = alpha * A * x + beta * y
template <int BS>
inline void bsr_spmv_old(LocalIndex n_rows, 
                     const LocalIndex* CFD_RESTRICT row_ptr,
                     const LocalIndex* CFD_RESTRICT cols, 
                     const double* CFD_RESTRICT values,
                     const double* CFD_RESTRICT x, 
                     double* CFD_RESTRICT y,
                     const double alpha, const double beta) {
    constexpr int BS2 = BS * BS;

    if (alpha == 0.0) {
        if (beta == 0.0) {
            std::fill_n(y, static_cast<std::size_t>(n_rows) * BS, 0.0);
        } else if (beta != 1.0) {
            const std::size_t total_size = static_cast<std::size_t>(n_rows) * BS;
            for (std::size_t i = 0; i < total_size; ++i) y[i] *= beta;
        }
        return;
    }

    for (LocalIndex i = 0; i < n_rows; ++i) {
        double acc[BS];
        for (int ii = 0; ii < BS; ++ii) acc[ii] = 0.0;

        for (LocalIndex k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * static_cast<std::size_t>(BS2);
            const double* CFD_RESTRICT xb = x + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(BS);

            for (int ii = 0; ii < BS; ++ii) {
                double s = 0.0;
                for (int jj = 0; jj < BS; ++jj) {
                    s += a[ii * BS + jj] * xb[jj];
                }
                acc[ii] += s;
            }
        }

        double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);

        if (alpha == 1.0 && beta == 0.0) {
            for (int ii = 0; ii < BS; ++ii) yb[ii] = acc[ii];
        } else if (beta == 1.0) {
            for (int ii = 0; ii < BS; ++ii) yb[ii] += alpha * acc[ii];
        } else if (beta == 0.0) {
            for (int ii = 0; ii < BS; ++ii) yb[ii] = alpha * acc[ii];
        } else {
            for (int ii = 0; ii < BS; ++ii) yb[ii] = alpha * acc[ii] + beta * yb[ii];
        }
    }
}

template <int BS>
inline void bsr_spmv(LocalIndex n_rows, 
                     const LocalIndex* CFD_RESTRICT row_ptr,
                     const LocalIndex* CFD_RESTRICT cols, 
                     const double* CFD_RESTRICT values,
                     const double* CFD_RESTRICT x, 
                     double* CFD_RESTRICT y,
                     const double alpha, const double beta) {
    constexpr int BS2 = BS * BS;

    if (alpha == 0.0) {
        if (beta == 0.0) {
            std::fill_n(y, static_cast<std::size_t>(n_rows) * BS, 0.0);
        } else if (beta != 1.0) {
            const std::size_t total_size = static_cast<std::size_t>(n_rows) * BS;
            for (std::size_t i = 0; i < total_size; ++i) y[i] *= beta;
        }
        return;
    }

    auto compute_row_acc = [&](LocalIndex i, double* CFD_RESTRICT acc) {
        const LocalIndex k1 = row_ptr[i + 1];
        for (int ii = 0; ii < BS; ++ii) acc[ii] = 0.0;
        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * static_cast<std::size_t>(BS2);
            const double* CFD_RESTRICT xb = x + static_cast<std::size_t>(cols[k]) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) {
                double s = 0.0;
                for (int jj = 0; jj < BS; ++jj) {
                    s += a[ii * BS + jj] * xb[jj];
                }
                acc[ii] += s;
            }
        }
    };

    if (alpha == 1.0 && beta == 0.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double acc[BS];
            compute_row_acc(i, acc);
            double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) yb[ii] = acc[ii];
        }
    } else if (beta == 1.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double acc[BS];
            compute_row_acc(i, acc);
            double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) yb[ii] += alpha * acc[ii];
        }
    } else if (beta == 0.0) {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double acc[BS];
            compute_row_acc(i, acc);
            double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) yb[ii] = alpha * acc[ii];
        }
    } else {
        for (LocalIndex i = 0; i < n_rows; ++i) {
            double acc[BS];
            compute_row_acc(i, acc);
            double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(BS);
            for (int ii = 0; ii < BS; ++ii) yb[ii] = alpha * acc[ii] + beta * yb[ii];
        }
    }
}

inline void bsr_spmv_dyn(LocalIndex n_rows, int bs, 
                         const LocalIndex* CFD_RESTRICT row_ptr,
                         const LocalIndex* CFD_RESTRICT cols, 
                         const double* CFD_RESTRICT values,
                         const double* CFD_RESTRICT x, 
                         double* CFD_RESTRICT y,
                         double alpha, double beta) {
    const std::size_t bs_sz = static_cast<std::size_t>(bs);
    const std::size_t bs2 = bs_sz * bs_sz;

    if (alpha == 0.0) {
        if (beta == 0.0) {
            std::fill_n(y, static_cast<std::size_t>(n_rows) * bs_sz, 0.0);
        } else if (beta != 1.0) {
            const std::size_t total_size = static_cast<std::size_t>(n_rows) * bs_sz;
            for (std::size_t i = 0; i < total_size; ++i) y[i] *= beta;
        }
        return;
    }

    for (LocalIndex i = 0; i < n_rows; ++i) {
        const LocalIndex k1 = row_ptr[i + 1];
        double* CFD_RESTRICT yb = y + static_cast<std::size_t>(i) * bs_sz;

        if (beta == 0.0) {
            for (int ii = 0; ii < bs; ++ii) yb[ii] = 0.0;
        } else if (beta != 1.0) {
            for (int ii = 0; ii < bs; ++ii) yb[ii] *= beta;
        }

        for (LocalIndex k = row_ptr[i]; k < k1; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * bs2;
            const double* CFD_RESTRICT xb = x + static_cast<std::size_t>(cols[k]) * bs_sz;

            for (int ii = 0; ii < bs; ++ii) {
                double s = 0.0;
                const std::size_t row_offset = static_cast<std::size_t>(ii) * bs_sz;
                for (int jj = 0; jj < bs; ++jj) {
                    s += a[row_offset + static_cast<std::size_t>(jj)] * xb[static_cast<std::size_t>(jj)];
                }
                yb[ii] += alpha * s;
            }
        }
    }
}

inline void bsr_spmv_dispatch(LocalIndex n_rows, int bs, 
                              const LocalIndex* CFD_RESTRICT row_ptr,
                              const LocalIndex* CFD_RESTRICT cols, 
                              const double* CFD_RESTRICT values,
                              const double* CFD_RESTRICT x, 
                              double* CFD_RESTRICT y,
                              double alpha, double beta) {
    switch (bs) {
        case 1: bsr_spmv<1>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 2: bsr_spmv<2>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 3: bsr_spmv<3>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 4: bsr_spmv<4>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 5: bsr_spmv<5>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 6: bsr_spmv<6>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 7: bsr_spmv<7>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        case 8: bsr_spmv<8>(n_rows, row_ptr, cols, values, x, y, alpha, beta); break;
        default: bsr_spmv_dyn(n_rows, bs, row_ptr, cols, values, x, y, alpha, beta); break;
    }
}





// aux LU factorization for matrix blocks
/// In-place LU with partial pivoting of a row-major bs x bs block.
/// Returns false for a (numerically) singular block.
inline bool lu_factor_block(double* CFD_RESTRICT a, const int bs, LocalIndex* CFD_RESTRICT pivots) {
    const std::size_t bs_sz = static_cast<std::size_t>(bs);

    for (int k = 0; k < bs; ++k) {
        const std::size_t k_offset = static_cast<std::size_t>(k) * bs_sz;

        int m = k;
        double mx = std::abs(a[k_offset + static_cast<std::size_t>(k)]);
        for (int i = k + 1; i < bs; ++i) {
            const double v = std::abs(a[static_cast<std::size_t>(i) * bs_sz + static_cast<std::size_t>(k)]);
            if (v > mx) {
                mx = v;
                m = i;
            }
        }
        
        pivots[k] = static_cast<LocalIndex>(m);
        if (mx == 0.0) return false;

        if (m != k) {
            const std::size_t m_offset = static_cast<std::size_t>(m) * bs_sz;
            for (int j = 0; j < bs; ++j) {
                std::swap(a[k_offset + static_cast<std::size_t>(j)],
                          a[m_offset + static_cast<std::size_t>(j)]);
            }
        }

        const double d = a[k_offset + static_cast<std::size_t>(k)];
        const double inv_d = 1.0 / d;

        for (int i = k + 1; i < bs; ++i) {
            const std::size_t i_offset = static_cast<std::size_t>(i) * bs_sz;
            
            const double l = a[i_offset + static_cast<std::size_t>(k)] * inv_d;
            a[i_offset + static_cast<std::size_t>(k)] = l;

            for (int j = k + 1; j < bs; ++j) {
                const std::size_t j_sz = static_cast<std::size_t>(j);
                a[i_offset + j_sz] -= l * a[k_offset + j_sz];
            }
        }
    }
    return true;
}

/// Solves (L U) b = rhs in place, compile-time block size BS.
template <int BS>
inline void lu_solve_block(const double* CFD_RESTRICT lu, 
                           const LocalIndex* CFD_RESTRICT piv,
                           double* CFD_RESTRICT b) {
    if constexpr (BS > 1) {
        for (int k = 0; k < BS; ++k) {
            if (piv[k] != k) std::swap(b[k], b[piv[k]]);
        }
    }

    // 2. (Forward substitution): L * y = b 
    for (int i = 1; i < BS; ++i) {
        double s = b[i];
        const int i_offset = i * BS;
        for (int j = 0; j < i; ++j) {
            s -= lu[i_offset + j] * b[j];
        }
        b[i] = s;
    }

    // 3. (Backward substitution): U * b = y 
    for (int i = BS - 1; i >= 0; --i) {
        double s = b[i];
        const int i_offset = i * BS;
        for (int j = i + 1; j < BS; ++j) {
            s -= lu[i_offset + j] * b[j];
        } 
        b[i] = s / lu[i_offset + i];
    }
}

/// Same for a runtime block size.
inline void lu_solve_block_dyn(const int bs, 
                               const double* CFD_RESTRICT lu,
                               const LocalIndex* CFD_RESTRICT piv, 
                               double* CFD_RESTRICT b) {
    const std::size_t bs_sz = static_cast<std::size_t>(bs);

    for (int k = 0; k < bs; ++k) {
        if (piv[k] != k) std::swap(b[k], b[piv[k]]);
    }

    for (int i = 1; i < bs; ++i) {
        double s = b[i];
        const std::size_t i_offset = static_cast<std::size_t>(i) * bs_sz;
        for (int j = 0; j < i; ++j) {
            s -= lu[i_offset + static_cast<std::size_t>(j)] * b[j];
        }
        b[i] = s;
    }

    for (int i = bs - 1; i >= 0; --i) {
        double s = b[i];
        const std::size_t i_offset = static_cast<std::size_t>(i) * bs_sz;
        for (int j = i + 1; j < bs; ++j) {
            s -= lu[i_offset + static_cast<std::size_t>(j)] * b[j];
        }
        b[i] = s / lu[i_offset + static_cast<std::size_t>(i)];
    }
}




//block symmetric Gauss-Seidel sweeps
template <int BS>
inline void bsr_sgs_sweep(const LocalIndex n_rows, bool forward, const LocalIndex* CFD_RESTRICT row_ptr,
                          const LocalIndex* CFD_RESTRICT cols, const double* CFD_RESTRICT values,
                          const LocalIndex* CFD_RESTRICT diag_idx, const double* CFD_RESTRICT diag_lu,
                          const LocalIndex* CFD_RESTRICT diag_piv, const double* CFD_RESTRICT r,
                          double* CFD_RESTRICT z) {
    constexpr int BS2 = BS * BS;
    for (LocalIndex cnt = 0; cnt < n_rows; ++cnt) {
        const LocalIndex i = forward ? cnt : static_cast<LocalIndex>(n_rows - 1 - cnt);
        const LocalIndex d = diag_idx[i];
        const LocalIndex k_start = row_ptr[i];
        const LocalIndex k_end   = row_ptr[i + 1];

        const double* CFD_RESTRICT rb = r + static_cast<std::size_t>(i) * BS;

        double s[BS];
        for (int ii = 0; ii < BS; ++ii) s[ii] = rb[ii];

        for (LocalIndex k = k_start; k < d; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * BS2;
            const double* CFD_RESTRICT zb = z + static_cast<std::size_t>(cols[k]) * BS;
            
            for (int ii = 0; ii < BS; ++ii) {
                double t = 0.0;
                for (int jj = 0; jj < BS; ++jj) t += a[ii * BS + jj] * zb[jj];
                s[ii] -= t;
            }
        }

        for (LocalIndex k = d + 1; k < k_end; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * BS2;
            const double* CFD_RESTRICT zb = z + static_cast<std::size_t>(cols[k]) * BS;
            
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
    const std::size_t bs_sz = static_cast<std::size_t>(bs);
    const std::size_t bs2 = bs_sz * bs_sz;
    
    static thread_local std::vector<double> s_buf;
    if (s_buf.size() < bs_sz) {
        s_buf.resize(bs_sz);
    }
    double* s = s_buf.data();

    for (LocalIndex cnt = 0; cnt < n_rows; ++cnt) {
        const LocalIndex i = forward ? cnt : static_cast<LocalIndex>(n_rows - 1 - cnt);
        const LocalIndex d = diag_idx[i];
        const LocalIndex k_start = row_ptr[i];
        const LocalIndex k_end   = row_ptr[i + 1];

        const std::size_t i_offset_bs = static_cast<std::size_t>(i) * bs_sz;
        const double* CFD_RESTRICT rb = r + i_offset_bs;

        for (int ii = 0; ii < bs; ++ii) s[ii] = rb[ii];

        for (LocalIndex k = k_start; k < d; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * bs2;
            const double* CFD_RESTRICT zb = z + static_cast<std::size_t>(cols[k]) * bs_sz;
            
            for (int ii = 0; ii < bs; ++ii) {
                double t = 0.0;
                const std::size_t row_offset = static_cast<std::size_t>(ii) * bs_sz;
                for (int jj = 0; jj < bs; ++jj) {
                    t += a[row_offset + static_cast<std::size_t>(jj)] * zb[jj];
                }
                s[ii] -= t;
            }
        }

        for (LocalIndex k = d + 1; k < k_end; ++k) {
            const double* CFD_RESTRICT a = values + static_cast<std::size_t>(k) * bs2;
            const double* CFD_RESTRICT zb = z + static_cast<std::size_t>(cols[k]) * bs_sz;
            
            for (int ii = 0; ii < bs; ++ii) {
                double t = 0.0;
                const std::size_t row_offset = static_cast<std::size_t>(ii) * bs_sz;
                for (int jj = 0; jj < bs; ++jj) {
                    t += a[row_offset + static_cast<std::size_t>(jj)] * zb[jj];
                }
                s[ii] -= t;
            }
        }

        lu_solve_block_dyn(bs, diag_lu + static_cast<std::size_t>(i) * bs2,
                           diag_piv + i_offset_bs, s);

        double* CFD_RESTRICT zi = z + i_offset_bs;
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

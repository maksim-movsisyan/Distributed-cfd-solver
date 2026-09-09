#pragma once

#include <mpi.h>

#include <memory>
#include <vector>
#include <cstring>


#include "cfd/core/types.hpp"

namespace cfd::linalg {

/**
 * @class VectorLayout
 * @brief Row distribution + ghost exchange pattern shared by matrices and
 * vectors of a distributed linear system.
 *
 * Rows are block-distributed with contiguous ownership: rank r owns the global
 * rows [localBegin(), localBegin() + localSize()). A rank may additionally
 * reference rows owned by others ("ghost rows"); ghost slots are appended
 * after the owned block, so a distributed vector is one contiguous array
 * [owned values | ghost values] and every hot kernel (SpMV, Gauss-Seidel
 * sweeps) works with plain local indices and pure local-memory access.
 *
 * The communication plan is derived purely from matrix sparsity (which global
 * columns a rank references), so the module knows nothing about meshes or
 * application-level halo exchangers.
 *
 * Layouts share their communication metadata via a shared pointer, but each 
 * copy has its own local scratch buffers for communication. This allows 
 * distinct vectors with their own Layout copies to perform asynchronous 
 * updates concurrently without data races.
 */
class VectorLayout {
public:
    // Empty layout on MPI_COMM_SELF (serial fallback).
    VectorLayout() : plan_(std::make_shared<Plan>()) {}

    // Contiguous ownership of `n_local` of `n_global` rows; the per-rank
    // start offset comes from a prefix sum (collective: MPI_Exscan).
    VectorLayout(MPI_Comm comm, GlobalIndex n_global, LocalIndex n_local);

    // Copy: share Plan, copy buffer data, fresh requests
    VectorLayout(const VectorLayout& other) 
        : plan_(other.plan_), 
          cached_bs_(other.cached_bs_) {
        send_buf_.resize(other.send_buf_.size());
        recv_buf_.resize(other.recv_buf_.size());
        requests_.clear();
    }
    VectorLayout& operator=(const VectorLayout& other) {
        if (this != &other) {
            plan_ = other.plan_;
            send_buf_.resize(other.send_buf_.size());
            recv_buf_.resize(other.recv_buf_.size());
            cached_bs_ = other.cached_bs_;
            requests_.clear();
        }
        return *this;
    }

    // Move: take ownership of everything
    VectorLayout(VectorLayout&&) noexcept = default;
    VectorLayout& operator=(VectorLayout&&) noexcept = default;

    // Auxiliary getters
    MPI_Comm comm() const { return plan_->comm; }
    int rank() const { return plan_->rank; }
    int nprocs() const { return plan_->nprocs; }

    GlobalIndex globalSize() const { return plan_->n_global; }
    LocalIndex localSize() const { return plan_->n_local; }
    GlobalIndex localBegin() const { return plan_->begin; }

    LocalIndex ghostSize() const { return static_cast<LocalIndex>(plan_->ghosts.size()); }
    /// Sorted, unique global ids of the ghost rows.
    const std::vector<GlobalIndex>& ghostGlobalIds() const { return plan_->ghosts; }

    /// Local slot of a global row (owned block first, ghosts after);
    /// kInvalidLocalIndex when the row is neither owned nor a declared ghost.
    /// Setup-time helper — hot kernels use precomputed local indices.
    LocalIndex localIndex(GlobalIndex global_row) const;

    /// True when both layouts describe the same distribution and ghost set
    /// (pointer-equal fast path, semantic fallback).
    bool compatibleWith(const VectorLayout& other) const;

    /**
     * @brief Declares the ghost rows this rank reads and builds the
     * point-to-point exchange plan (collective on the layout communicator).
     *
     * Called by matrix assembly from the set of foreign columns it references;
     * a layout without ghosts is valid as-is (no call needed).
     * @param sorted_unique_ghost_ids ascending, unique, foreign global rows.
     */
    void setGhosts(const std::vector<GlobalIndex>& sorted_unique_ghost_ids);

    /**
     * @brief Refreshes the ghost entries of `values` — a contiguous array of
     * (localSize() + ghostSize()) * block_size doubles — from their owners.
     *
     * Non-blocking send/recv + wait-all: one synchronous MPI round per call.
     * Modifies internal scratch buffers (not thread-safe for the same instance).
     */
    void updateGhosts(double* values, int block_size);

    /**
     * @brief Refreshes the ghost entries of `values` — a contiguous array of
     * (localSize() + ghostSize()) * block_size doubles — from their owners.
     *
     * Non-blocking send/recv: posts Isend/Irecv and packs data into internal buffers.
     * Modifies internal scratch buffers (not thread-safe for the same instance).
     */
    void updateGhosts_start(double* values, int block_size);

    /**
     * @brief Completes the ghost exchange started by updateGhosts_start().
     * 
     * Unpacks received data from internal buffers into the ghost slots of `values`.
     */
    void updateGhosts_end(double* values, int block_size);

private:
    struct Plan {
        MPI_Comm comm = MPI_COMM_SELF;
        bool owns_comm = false;
        int rank = 0;
        int nprocs = 1;

        GlobalIndex n_global = 0;
        LocalIndex n_local = 0;
        GlobalIndex begin = 0;

        std::vector<GlobalIndex> ghosts;      // sorted unique foreign rows
        std::vector<GlobalIndex> all_begins;  // per-rank ownership starts

        // Exchange plan, grouped by neighbour rank (ascending):
        //   send — my owned rows that other ranks ghost,
        //   recv — ghost slots of mine that other ranks own.
        std::vector<int> send_ranks, recv_ranks;
        std::vector<int> send_counts, recv_counts;  // rows per neighbour
        std::vector<int> send_displ, recv_displ;    // offsets into the idx arrays
        std::vector<LocalIndex> send_idx;           // owned local slots
        std::vector<LocalIndex> recv_idx;           // ghost local slots (>= n_local)

        ~Plan() {
            if (owns_comm && comm != MPI_COMM_SELF && comm != MPI_COMM_NULL) {
                MPI_Comm_free(&comm);
            }
        }
    };
 
    std::shared_ptr<Plan> plan_;  // shared: copies of a layout share the plan

     // Scratch for updateGhosts (sized for the block size in use).
    // Mutated during exchange operations, isolating state between Layout copies.
    std::vector<double> send_buf_, recv_buf_;
    std::vector<MPI_Request> requests_;
    int cached_bs_ = 0;

    void resize_bufs(int block_size) {
        const std::size_t tot_send = plan_->send_idx.size();
        const std::size_t tot_recv = plan_->recv_idx.size();
        
        send_buf_.resize(tot_send * static_cast<std::size_t>(block_size));
        recv_buf_.resize(tot_recv * static_cast<std::size_t>(block_size));
        
        cached_bs_ = block_size;
    }

    void pack_values(const double* CFD_RESTRICT values) {
        const std::size_t n_send = plan_->send_idx.data() ? plan_->send_idx.size() : 0;
        if (n_send == 0) return;

        const LocalIndex* CFD_RESTRICT src_idx = plan_->send_idx.data();
        double* CFD_RESTRICT dst_buf = send_buf_.data();
        const std::size_t bs = static_cast<std::size_t>(cached_bs_);

        if (bs == 1) {
            for (std::size_t i = 0; i < n_send; ++i) {
                dst_buf[i] = values[src_idx[i]];
            }
        } else if (bs == 3) {
            for (std::size_t i = 0; i < n_send; ++i) {
                const std::size_t src_offset = static_cast<std::size_t>(src_idx[i]) * 3;
                const std::size_t dst_offset = i * 3;
                dst_buf[dst_offset + 0] = values[src_offset + 0];
                dst_buf[dst_offset + 1] = values[src_offset + 1];
                dst_buf[dst_offset + 2] = values[src_offset + 2];
            }
        } else if (bs == 5) {
            for (std::size_t i = 0; i < n_send; ++i) {
                const std::size_t src_offset = static_cast<std::size_t>(src_idx[i]) * 5;
                const std::size_t dst_offset = i * 5;
                dst_buf[dst_offset + 0] = values[src_offset + 0];
                dst_buf[dst_offset + 1] = values[src_offset + 1];
                dst_buf[dst_offset + 2] = values[src_offset + 2];
                dst_buf[dst_offset + 3] = values[src_offset + 3];
                dst_buf[dst_offset + 4] = values[src_offset + 4];
            }
        } else {
            for (std::size_t i = 0; i < n_send; ++i) {
                const std::size_t src_offset = static_cast<std::size_t>(src_idx[i]) * bs;
                const std::size_t dst_offset = i * bs;
                

                for (std::size_t b = 0; b < bs; ++b) {
                    dst_buf[dst_offset + b] = values[src_offset + b];
                }
            }
        }
    }

    void post_sr(int tag = 4317) {
        const std::size_t ns = plan_->send_ranks.size();
        const std::size_t nr = plan_->recv_ranks.size();
        const std::size_t bs = static_cast<std::size_t>(cached_bs_);
 
        if (requests_.size() != ns + nr) {
            requests_.resize(ns + nr);
        }

        std::size_t rq = 0;
        

        for (size_t r = 0; r < nr; ++r) {
            const size_t offset = static_cast<size_t>(plan_->recv_displ[r]) * bs;
            const int count = plan_->recv_counts[r] * cached_bs_;
            
            MPI_Irecv(recv_buf_.data() + offset, count, MPI_DOUBLE, 
                    plan_->recv_ranks[r], tag, plan_->comm, &requests_[rq++]);
        }
        
        for (size_t s = 0; s < ns; ++s) {
            const size_t offset = static_cast<size_t>(plan_->send_displ[s]) * bs;
            const int count = plan_->send_counts[s] * cached_bs_;
            
            MPI_Isend(send_buf_.data() + offset, count, MPI_DOUBLE, 
                    plan_->send_ranks[s], tag, plan_->comm, &requests_[rq++]);
        }
    }

    void wait_and_unpack(double* CFD_RESTRICT values, int block_size) {
        const std::size_t ns = plan_->send_ranks.size();
        const std::size_t nr = plan_->recv_ranks.size();
        const std::size_t bs = static_cast<std::size_t>(block_size);
        if (ns + nr == 0) return;

        MPI_Waitall(static_cast<int>(ns + nr), requests_.data(), MPI_STATUSES_IGNORE);

        const std::size_t n_recv = plan_->recv_idx.size();
        if (n_recv == 0) return;

        const double* CFD_RESTRICT src_buf = recv_buf_.data();
        const LocalIndex* CFD_RESTRICT dst_idx = plan_->recv_idx.data();
     
        if (dst_idx[0] == plan_->n_local && dst_idx[n_recv - 1] == plan_->n_local + static_cast<LocalIndex>(n_recv) - 1) {
            std::memcpy(values + static_cast<std::size_t>(plan_->n_local) * bs, 
                        src_buf, n_recv * bs * sizeof(double));
            return;  
        }

        if (block_size == 1) {
            for (std::size_t i = 0; i < n_recv; ++i) {
                values[dst_idx[i]] = src_buf[i];
            }
        }
        else if (block_size == 3) {
            for (std::size_t i = 0; i < n_recv; ++i) {
                const std::size_t dst_offset = static_cast<std::size_t>(dst_idx[i]) * 3;
                const std::size_t src_offset = i * 3;
                values[dst_offset + 0] = src_buf[src_offset + 0];
                values[dst_offset + 1] = src_buf[src_offset + 1];
                values[dst_offset + 2] = src_buf[src_offset + 2];
            }
        }
        else if (block_size == 5) {
            for (std::size_t i = 0; i < n_recv; ++i) {
                const std::size_t dst_offset = static_cast<std::size_t>(dst_idx[i]) * 5;
                const std::size_t src_offset = i * 5;
                values[dst_offset + 0] = src_buf[src_offset + 0];
                values[dst_offset + 1] = src_buf[src_offset + 1];
                values[dst_offset + 2] = src_buf[src_offset + 2];
                values[dst_offset + 3] = src_buf[src_offset + 3];
                values[dst_offset + 4] = src_buf[src_offset + 4];
            }
        }
        else {
            for (std::size_t i = 0; i < n_recv; ++i) {
                const std::size_t dst_offset = static_cast<std::size_t>(dst_idx[i]) * bs;
                const std::size_t src_offset = i * bs;
                
                for (std::size_t b = 0; b < bs; ++b) {
                    values[dst_offset + b] = src_buf[src_offset + b];
                }
            }
        }
    }

};

}  // namespace cfd::linalg

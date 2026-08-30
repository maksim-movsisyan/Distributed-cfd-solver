#pragma once

#include <mpi.h>

#include <cstddef>
#include <span>
#include <vector>

#include "cfd/mesh/localmesh.hpp"

namespace cfd::solver::halo {

/**
 * @class HaloExchanger
 * @brief Zero-allocation aggregated MPI halo exchanger for registered cell fields.
 *
 * Packs all registered variables per phase into a SINGLE contiguous message
 * per partition neighbour, minimizing network latency and MPI overhead.
 */
class HaloExchanger {
public:
    HaloExchanger(const mesh::MeshPart& mp, MPI_Comm comm);
    ~HaloExchanger() = default;

    // Non-copyable, non-movable (maintains stable MPI buffers and internal pointers)
    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;
    HaloExchanger(HaloExchanger&&) = delete;
    HaloExchanger& operator=(HaloExchanger&&) = delete;

    // --- Registration (Initialization Time Only) -----------------------------

    /**
     * @brief Registers standard cell fields (1 double per cell).
     * @param fields Pointers to contiguous cell arrays of size >= n_cells.
     */
    void register_cell_fields(std::span<double* const> fields);

    /**
     * @brief Registers gradient components (3 planes per var) and limiter arrays.
     * @param grad_bases   Base pointers for each variable (dx is at base, dy at base + stride, dz at base + 2*stride).
     * @param plane_stride Stride between gradient planes in doubles (usually allocated cell count).
     * @param limiters     Optional limiter field pointers (1 double per cell). Can be empty.
     */
    void register_grad_limiters(std::span<double* const> grad_bases,
                                std::size_t plane_stride,
                                std::span<double* const> limiters);

    // --- Primary Fields Communication Phase ----------------------------------
    void exchange_fields();
    void start_exchange_fields();
    void finish_exchange_fields();

    // --- Gradients + Limiters Communication Phase ----------------------------
    void exchange_grad_limiters();
    void start_exchange_grad_limiters();
    void finish_exchange_grad_limiters();

    // --- Diagnostics ---------------------------------------------------------
    [[nodiscard]] std::size_t num_registered_fields() const noexcept { return fields_phase_.ptrs.size(); }
    [[nodiscard]] std::size_t num_registered_grads_and_limiters() const noexcept { return grads_phase_.ptrs.size(); }

private:
    struct PhaseData {
        std::vector<double*> ptrs;             // Flat array of SoA field pointers
        std::vector<double>  send_buf;
        std::vector<double>  recv_buf;
        std::vector<MPI_Request> requests;     // [0..nnb) - Recv, [nnb..2*nnb) - Send
        int tag = 0;
    };

    void resize_phase(PhaseData& phase, const char* phase_name);
    void post_irecvs(PhaseData& phase);
    void pack_and_isend(PhaseData& phase);
    void unpack_ghosts(PhaseData& phase);

    MPI_Comm comm_{MPI_COMM_NULL};
    const mesh::MeshPart& mp_;
    std::vector<int> nb_ranks_;

    PhaseData fields_phase_;
    PhaseData grads_phase_;
};

} // namespace cfd::solver::halo
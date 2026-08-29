// One-hop non-blocking MPI halo exchanger for SoA field views, gradients, and limiters.
#pragma once

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::halo {

inline constexpr std::size_t kGradPayload = 20; // 15 gradient doubles (3 dims x 5 vars) + 5 limiter doubles

/**
 * @class HaloExchanger
 * @brief Zero-allocation MPI halo exchanger for cell fields, gradients, and limiters.
 * 
 * Manages non-blocking point-to-point exchanges with partition neighbours.
 * Data is unpacked directly into the halo ghost region [n_own_cells, n_cells).
 */
class HaloExchanger {
public:
    HaloExchanger(const mesh::MeshPart& mp, MPI_Comm comm);

    // --- Primary Fields Exchange (5 variables: Conservative or Primitive) ---
    // --- All-in-one blocking exchange ---
    void exchange(fields::ConservativeView<double> u);
    void exchange(fields::PrimitiveView<double> q);

    // --- Non-blocking split exchange (for computation / communication overlap) ---
    void start_exchange(fields::ConservativeView<double> u);
    void finish_exchange(fields::ConservativeView<double> u);

    void start_exchange(fields::PrimitiveView<double> q);
    void finish_exchange(fields::PrimitiveView<double> q);

    // --- 2nd-Order Gradients + Limiters Combined Exchange (20 variables) ---
    // --- All-in-one blocking exchange ---
    void exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                               fields::PrimitiveView<double> phi);

    // --- Non-blocking split exchange (for computation / communication overlap) ---
    void start_exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                                     fields::PrimitiveView<double> phi);
    void finish_exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                                      fields::PrimitiveView<double> phi);

private:
    void post_field_irecvs();
    void post_grad_irecvs();

    template <typename View>
    void pack_and_isend_fields(const View& v);

    template <typename View>
    void unpack_field_ghosts(View& v);

    void pack_and_isend_grad(fields::PrimitiveGradView<double> grad,
                             fields::PrimitiveView<double> phi);

    void unpack_grad_ghosts(fields::PrimitiveGradView<double> grad,
                            fields::PrimitiveView<double> phi);

    MPI_Comm comm_{MPI_COMM_NULL};
    const mesh::MeshPart& mp_;
    std::vector<int> nb_ranks_;

    // MPI Request handles
    std::vector<MPI_Request> field_requests_;
    std::vector<MPI_Request> grad_requests_;

    // Buffers for 5-variable field exchanges
    std::vector<double> send_buf_;
    std::vector<double> recv_buf_;

    // Buffers for 20-variable gradient+limiter exchanges
    std::vector<double> grad_send_buf_;
    std::vector<double> grad_recv_buf_;
};

} // namespace cfd::solver::halo
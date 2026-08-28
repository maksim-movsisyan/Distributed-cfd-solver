// One-hop non-blocking halo exchange for 5-variable SoA field views.
#pragma once

#include <mpi.h>

#include <vector>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/solver/fields/fields_view.hpp"

namespace cfd::solver::halo {

/**
 * @class HaloExchanger
 * @brief Zero-allocation MPI halo exchanger for SoA cell fields.
 * 
 * Packs send-list cells, exchanges ghost data with partition neighbours via
 * non-blocking point-to-point routines, and unpacks directly into the ghost region
 * [n_own_cells, n_cells).
 */
class HaloExchanger {
public:
    HaloExchanger(const mesh::MeshPart& mp, MPI_Comm comm);

    // --- All-in-one blocking exchange ---
    void exchange(fields::ConservativeView<double> u);
    void exchange(fields::PrimitiveView<double> q);

    // --- Non-blocking split exchange (for computation / communication overlap) ---
    void start_exchange(fields::ConservativeView<double> u);
    void finish_exchange(fields::ConservativeView<double> u);

    void start_exchange(fields::PrimitiveView<double> q);
    void finish_exchange(fields::PrimitiveView<double> q);

private:
    template <typename View>
    void pack_and_isend(const View& v);

    template <typename View>
    void unpack_ghosts(View& v);

    void post_irecvs();

    MPI_Comm comm_{MPI_COMM_NULL};
    const mesh::MeshPart& mp_;
    std::vector<int> nb_ranks_;
    std::vector<MPI_Request> requests_;

    // Continuous cell-major interleaved communication buffers
    std::vector<double> send_buf_;
    std::vector<double> recv_buf_;
};

} // namespace cfd::solver::halo
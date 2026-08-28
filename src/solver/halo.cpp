#include "cfd/solver/halo.hpp"

#include <cstddef>
#include <vector>

#include "cfd/mpi/log.hpp"
#include "cfd/solver/eos/state_conversions.hpp"

namespace cfd::solver::halo {

namespace {
// Unique MPI tag dedicated to halo field exchanges
constexpr int kHaloTag = 4242;
} // anonymous namespace

HaloExchanger::HaloExchanger(const mesh::MeshPart& mp, const MPI_Comm comm)
    : comm_(comm), mp_(mp), nb_ranks_(mp.nb_ranks) {
    const std::size_t nnb = nb_ranks_.size();
    if (nnb == 0) {
        return;
    }

    const std::size_t total_send = mp.send_offsets.empty() ? 0 : static_cast<std::size_t>(mp.send_offsets.back());
    const std::size_t total_recv = mp.recv_offsets.empty() ? 0 : static_cast<std::size_t>(mp.recv_offsets.back());

    constexpr std::size_t max_doubles = 0x7FFFFFFFull / eos::kNumVars;
    if (total_send > max_doubles || total_recv > max_doubles) {
        mpi::fatal(comm_, "HaloExchanger: per-neighbour cell counts exceed MPI integer limits");
    }

    send_buf_.assign(total_send * eos::kNumVars, 0.0);
    recv_buf_.assign(total_recv * eos::kNumVars, 0.0);
    requests_.assign(2 * nnb, MPI_REQUEST_NULL);
}

void HaloExchanger::post_irecvs() {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const int count = static_cast<int>((ro[k + 1] - ro[k]) * eos::kNumVars);
        MPI_Irecv(recv_buf_.data() + static_cast<std::ptrdiff_t>(ro[k]) * eos::kNumVars,
                  count, MPI_DOUBLE, nb_ranks_[k], kHaloTag, comm_, &requests_[k]);
    }
}

template <typename View>
void HaloExchanger::pack_and_isend(const View& v) {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT so = mp_.send_offsets.data();
    const LocalIndex* CFD_RESTRICT sv = mp_.send_owned_local.data();

    // Extract raw pointers from SoA View
    const double* CFD_RESTRICT f0 = nullptr;
    const double* CFD_RESTRICT f1 = nullptr;
    const double* CFD_RESTRICT f2 = nullptr;
    const double* CFD_RESTRICT f3 = nullptr;
    const double* CFD_RESTRICT f4 = nullptr;

    if constexpr (std::is_same_v<View, fields::ConservativeView<double>>) {
        f0 = v.rho;  f1 = v.rhou; f2 = v.rhov; f3 = v.rhow; f4 = v.rhoE;
    } else {
        f0 = v.prs;  f1 = v.vx;   f2 = v.vy;   f3 = v.vz;   f4 = v.tmp;
    }

    for (std::size_t k = 0; k < nnb; ++k) {
        double* CFD_RESTRICT dst = send_buf_.data() + static_cast<std::ptrdiff_t>(so[k]) * eos::kNumVars;

        for (LocalIndex i = so[k]; i < so[k + 1]; ++i) {
            const LocalIndex c = sv[i];
            dst[0] = f0[c];
            dst[1] = f1[c];
            dst[2] = f2[c];
            dst[3] = f3[c];
            dst[4] = f4[c];
            dst += eos::kNumVars;
        }

        const int count = static_cast<int>((so[k + 1] - so[k]) * eos::kNumVars);
        MPI_Isend(send_buf_.data() + static_cast<std::ptrdiff_t>(so[k]) * eos::kNumVars,
                  count, MPI_DOUBLE, nb_ranks_[k], kHaloTag, comm_, &requests_[nnb + k]);
    }
}

template <typename View>
void HaloExchanger::unpack_ghosts(View& v) {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();
    const LocalIndex* CFD_RESTRICT rv = mp_.recv_ghost_local.data();

    double* CFD_RESTRICT f0 = nullptr;
    double* CFD_RESTRICT f1 = nullptr;
    double* CFD_RESTRICT f2 = nullptr;
    double* CFD_RESTRICT f3 = nullptr;
    double* CFD_RESTRICT f4 = nullptr;

    if constexpr (std::is_same_v<View, fields::ConservativeView<double>>) {
        f0 = v.rho;  f1 = v.rhou; f2 = v.rhov; f3 = v.rhow; f4 = v.rhoE;
    } else {
        f0 = v.prs;  f1 = v.vx;   f2 = v.vy;   f3 = v.vz;   f4 = v.tmp;
    }

    for (std::size_t k = 0; k < nnb; ++k) {
        const double* CFD_RESTRICT src = recv_buf_.data() + static_cast<std::ptrdiff_t>(ro[k]) * eos::kNumVars;

        for (LocalIndex i = ro[k]; i < ro[k + 1]; ++i) {
            const LocalIndex c = rv[i];
            f0[c] = src[0];
            f1[c] = src[1];
            f2[c] = src[2];
            f3[c] = src[3];
            f4[c] = src[4];
            src += eos::kNumVars;
        }
    }
}

void HaloExchanger::start_exchange(fields::ConservativeView<double> u) {
    if (nb_ranks_.empty()) return;
    post_irecvs();
    pack_and_isend(u);
}

void HaloExchanger::finish_exchange(fields::ConservativeView<double> u) {
    if (nb_ranks_.empty()) return;
    MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
    unpack_ghosts(u);
}

void HaloExchanger::start_exchange(fields::PrimitiveView<double> q) {
    if (nb_ranks_.empty()) return;
    post_irecvs();
    pack_and_isend(q);
}

void HaloExchanger::finish_exchange(fields::PrimitiveView<double> q) {
    if (nb_ranks_.empty()) return;
    MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
    unpack_ghosts(q);
}

void HaloExchanger::exchange(fields::ConservativeView<double> u) {
    start_exchange(u);
    finish_exchange(u);
}

void HaloExchanger::exchange(fields::PrimitiveView<double> q) {
    start_exchange(q);
    finish_exchange(q);
}

} // namespace cfd::solver::halo
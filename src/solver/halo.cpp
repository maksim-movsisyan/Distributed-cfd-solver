#include "cfd/solver/halo.hpp"

#include <cstddef>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/solver/eos/state_conversions.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::solver::halo {

namespace {
// Unique MPI tags for distinct communication channels
constexpr int kFieldTag       = 4242;
constexpr int kGradLimiterTag = 4243;
} // anonymous namespace

HaloExchanger::HaloExchanger(const mesh::MeshPart& mp, const MPI_Comm comm)
    : comm_(comm), mp_(mp), nb_ranks_(mp.nb_ranks) {
    const std::size_t nnb = nb_ranks_.size();
    if (nnb == 0) {
        return;
    }

    const std::size_t total_send = mp.send_offsets.empty() ? 0 : static_cast<std::size_t>(mp.send_offsets.back());
    const std::size_t total_recv = mp.recv_offsets.empty() ? 0 : static_cast<std::size_t>(mp.recv_offsets.back());

    constexpr std::size_t max_doubles = 0x7FFFFFFFull / kGradPayload;
    if (total_send > max_doubles || total_recv > max_doubles) {
        mpi::fatal(comm_, "HaloExchanger: cell count per neighbour exceeds MPI integer limit");
    }

    // Allocate 5-variable field buffers
    send_buf_.assign(total_send * eos::kNumVars, 0.0);
    recv_buf_.assign(total_recv * eos::kNumVars, 0.0);
    field_requests_.assign(2 * nnb, MPI_REQUEST_NULL);

    // Allocate 20-variable gradient + limiter buffers
    grad_send_buf_.assign(total_send * kGradPayload, 0.0);
    grad_recv_buf_.assign(total_recv * kGradPayload, 0.0);
    grad_requests_.assign(2 * nnb, MPI_REQUEST_NULL);
}

// ============================================================================
// --- Primary Fields Communication (5 variables) -----------------------------
// ============================================================================

void HaloExchanger::post_field_irecvs() {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const int count = static_cast<int>((ro[k + 1] - ro[k]) * eos::kNumVars);
        MPI_Irecv(recv_buf_.data() + static_cast<std::ptrdiff_t>(ro[k]) * eos::kNumVars,
                  count, MPI_DOUBLE, nb_ranks_[k], kFieldTag, comm_, &field_requests_[k]);
    }
}

template <typename View>
void HaloExchanger::pack_and_isend_fields(const View& v) {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT so = mp_.send_offsets.data();
    const LocalIndex* CFD_RESTRICT sv = mp_.send_owned_local.data();

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
                  count, MPI_DOUBLE, nb_ranks_[k], kFieldTag, comm_, &field_requests_[nnb + k]);
    }
}

template <typename View>
void HaloExchanger::unpack_field_ghosts(View& v) {
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
    post_field_irecvs();
    pack_and_isend_fields(u);
}

void HaloExchanger::finish_exchange(fields::ConservativeView<double> u) {
    if (nb_ranks_.empty()) return;
    MPI_Waitall(static_cast<int>(field_requests_.size()), field_requests_.data(), MPI_STATUSES_IGNORE);
    unpack_field_ghosts(u);
}

void HaloExchanger::start_exchange(fields::PrimitiveView<double> q) {
    if (nb_ranks_.empty()) return;
    post_field_irecvs();
    pack_and_isend_fields(q);
}

void HaloExchanger::finish_exchange(fields::PrimitiveView<double> q) {
    if (nb_ranks_.empty()) return;
    MPI_Waitall(static_cast<int>(field_requests_.size()), field_requests_.data(), MPI_STATUSES_IGNORE);
    unpack_field_ghosts(q);
}

void HaloExchanger::exchange(fields::ConservativeView<double> u) {
    start_exchange(u);
    finish_exchange(u);
}

void HaloExchanger::exchange(fields::PrimitiveView<double> q) {
    start_exchange(q);
    finish_exchange(q);
}

// ============================================================================
// --- Gradients + Limiters Communication (20 variables) ----------------------
// ============================================================================

void HaloExchanger::post_grad_irecvs() {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const int count = static_cast<int>(static_cast<std::size_t>(ro[k + 1] - ro[k]) * kGradPayload);
        MPI_Irecv(grad_recv_buf_.data() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ro[k]) * kGradPayload),
                  count, MPI_DOUBLE, nb_ranks_[k], kGradLimiterTag, comm_, &grad_requests_[k]);
    }
}

void HaloExchanger::pack_and_isend_grad(fields::PrimitiveGradView<double> grad,
                                       fields::PrimitiveView<double> phi) {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT so = mp_.send_offsets.data();
    const LocalIndex* CFD_RESTRICT sv = mp_.send_owned_local.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        double* CFD_RESTRICT dst = grad_send_buf_.data() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(so[k]) * kGradPayload);

        for (LocalIndex i = so[k]; i < so[k + 1]; ++i) {
            const auto c = static_cast<std::size_t>(sv[i]);

            // 1. Pack 15 gradient components
            dst[0]  = grad.dprs_dx(c);
            dst[1]  = grad.dvx_dx(c);
            dst[2]  = grad.dvy_dx(c);
            dst[3]  = grad.dvz_dx(c);
            dst[4]  = grad.dtmp_dx(c);

            dst[5]  = grad.dprs_dy(c);
            dst[6]  = grad.dvx_dy(c);
            dst[7]  = grad.dvy_dy(c);
            dst[8]  = grad.dvz_dy(c);
            dst[9]  = grad.dtmp_dy(c);

            dst[10] = grad.dprs_dz(c);
            dst[11] = grad.dvx_dz(c);
            dst[12] = grad.dvy_dz(c);
            dst[13] = grad.dvz_dz(c);
            dst[14] = grad.dtmp_dz(c);

            // 2. Pack 5 limiter values
            dst[15] = phi.prs[c];
            dst[16] = phi.vx[c];
            dst[17] = phi.vy[c];
            dst[18] = phi.vz[c];
            dst[19] = phi.tmp[c];

            dst += kGradPayload;
        }

        const int count = static_cast<int>(static_cast<std::size_t>(so[k + 1] - so[k]) * kGradPayload);
        MPI_Isend(grad_send_buf_.data() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(so[k]) * kGradPayload),
                  count, MPI_DOUBLE, nb_ranks_[k], kGradLimiterTag, comm_, &grad_requests_[nnb + k]);
    }
}

void HaloExchanger::unpack_grad_ghosts(fields::PrimitiveGradView<double> grad,
                                      fields::PrimitiveView<double> phi) {
    const std::size_t nnb = nb_ranks_.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();
    const LocalIndex* CFD_RESTRICT rv = mp_.recv_ghost_local.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const double* CFD_RESTRICT src = grad_recv_buf_.data() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ro[k]) * kGradPayload);

        for (LocalIndex i = ro[k]; i < ro[k + 1]; ++i) {
            const auto c = static_cast<std::size_t>(rv[i]);

            // 1. Unpack 15 gradient components
            grad.dprs_dx(c) = src[0];
            grad.dvx_dx(c)  = src[1];
            grad.dvy_dx(c)  = src[2];
            grad.dvz_dx(c)  = src[3];
            grad.dtmp_dx(c) = src[4];

            grad.dprs_dy(c) = src[5];
            grad.dvx_dy(c)  = src[6];
            grad.dvy_dy(c)  = src[7];
            grad.dvz_dy(c)  = src[8];
            grad.dtmp_dy(c) = src[9];

            grad.dprs_dz(c) = src[10];
            grad.dvx_dz(c)  = src[11];
            grad.dvy_dz(c)  = src[12];
            grad.dvz_dz(c)  = src[13];
            grad.dtmp_dz(c) = src[14];

            // 2. Unpack 5 limiter values
            phi.prs[c] = src[15];
            phi.vx[c]  = src[16];
            phi.vy[c]  = src[17];
            phi.vz[c]  = src[18];
            phi.tmp[c] = src[19];

            src += kGradPayload;
        }
    }
}

void HaloExchanger::start_exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                                               fields::PrimitiveView<double> phi) {
    if (nb_ranks_.empty()) return;
    post_grad_irecvs();
    pack_and_isend_grad(grad, phi);
}

void HaloExchanger::finish_exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                                                fields::PrimitiveView<double> phi) {
    if (nb_ranks_.empty()) return;
    MPI_Waitall(static_cast<int>(grad_requests_.size()), grad_requests_.data(), MPI_STATUSES_IGNORE);
    unpack_grad_ghosts(grad, phi);
}

void HaloExchanger::exchange_grad_limiter(fields::PrimitiveGradView<double> grad,
                                         fields::PrimitiveView<double> phi) {
    start_exchange_grad_limiter(grad, phi);
    finish_exchange_grad_limiter(grad, phi);
}

} // namespace cfd::solver::halo
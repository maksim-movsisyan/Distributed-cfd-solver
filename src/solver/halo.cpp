#include "cfd/solver/halo.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <cstring>

#include "cfd/core/types.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::solver::halo {

namespace {

constexpr int kFieldTag       = 4242;
constexpr int kGradLimiterTag = 4243;
constexpr std::size_t kMaxStackVars = 64;
constexpr std::int64_t kMpiCountLimit = 0x7FFFFFFFLL;

// ============================================================================
// --- High-Performance Vectorized Packing / Unpacking Kernels ----------------
// ============================================================================

template <std::size_t FixedVars>
inline void pack_soa_to_aos_kernel(const LocalIndex* CFD_RESTRICT send_map,
                                   const std::size_t count,
                                   const double* const* CFD_RESTRICT fields,
                                   const std::size_t num_vars,
                                   double* CFD_RESTRICT dst) {
    if constexpr (FixedVars > 0) {
        // Compile-time fixed size: full loop unrolling & maximum SIMD utilization
        const double* CFD_RESTRICT local_fields[FixedVars];
        
        for (std::size_t v = 0; v < FixedVars; ++v) {
            local_fields[v] = fields[v];
        }

        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t c = static_cast<std::size_t>(send_map[i]);
            
            for (std::size_t v = 0; v < FixedVars; ++v) {
                dst[v] = local_fields[v][c];
            }
            dst += FixedVars;
        }
    } else {
        // Dynamic variable count fallback (turb, species, scalars)
        if (num_vars <= kMaxStackVars) {
            const double* CFD_RESTRICT local_fields[kMaxStackVars];
            for (std::size_t v = 0; v < num_vars; ++v) {
                local_fields[v] = fields[v];
            }

            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t c = static_cast<std::size_t>(send_map[i]);
                for (std::size_t v = 0; v < num_vars; ++v) {
                    dst[v] = local_fields[v][c];
                }
                dst += num_vars;
            }
        } else {
            // Large reaction mechanisms (>64 species)
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t c = static_cast<std::size_t>(send_map[i]);
                for (std::size_t v = 0; v < num_vars; ++v) {
                    dst[v] = fields[v][c];
                }
                dst += num_vars;
            }
        }
    }
}

template <std::size_t FixedVars>
inline void unpack_aos_to_soa_kernel(const LocalIndex* CFD_RESTRICT recv_map,
                                     const std::size_t count,
                                     const double* CFD_RESTRICT src,
                                     const std::size_t num_vars,
                                     double* const* CFD_RESTRICT fields) {
    if constexpr (FixedVars > 0) {
        double* CFD_RESTRICT local_fields[FixedVars];

        for (std::size_t v = 0; v < FixedVars; ++v) {
            local_fields[v] = fields[v];
        }

        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t c = static_cast<std::size_t>(recv_map[i]);

            for (std::size_t v = 0; v < FixedVars; ++v) {
                local_fields[v][c] = src[v];
            }
            src += FixedVars;
        }
    } else {
        if (num_vars <= kMaxStackVars) {
            double* CFD_RESTRICT local_fields[kMaxStackVars];
            for (std::size_t v = 0; v < num_vars; ++v) {
                local_fields[v] = fields[v];
            }

            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t c = static_cast<std::size_t>(recv_map[i]);
                for (std::size_t v = 0; v < num_vars; ++v) {
                    local_fields[v][c] = src[v];
                }
                src += num_vars;
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t c = static_cast<std::size_t>(recv_map[i]);
                for (std::size_t v = 0; v < num_vars; ++v) {
                    fields[v][c] = src[v];
                }
                src += num_vars;
            }
        }
    }
}

// Fast dispatcher based on payload size
inline void dispatch_pack(const LocalIndex* CFD_RESTRICT send_map,
                          const std::size_t count,
                          const double* const* CFD_RESTRICT fields,
                          const std::size_t num_vars,
                          double* CFD_RESTRICT dst) {
    switch (num_vars) {
        case 5:  pack_soa_to_aos_kernel<5>(send_map, count, fields, 5, dst); break;
        case 20: pack_soa_to_aos_kernel<20>(send_map, count, fields, 20, dst); break;
        default: pack_soa_to_aos_kernel<0>(send_map, count, fields, num_vars, dst); break;
    }
}

inline void dispatch_unpack(const LocalIndex* CFD_RESTRICT recv_map,
                            const std::size_t count,
                            const double* CFD_RESTRICT src,
                            const std::size_t num_vars,
                            double* const* CFD_RESTRICT fields) {
    switch (num_vars) {
        case 5:  unpack_aos_to_soa_kernel<5>(recv_map, count, src, 5, fields); break;
        case 20: unpack_aos_to_soa_kernel<20>(recv_map, count, src, 20, fields); break;
        default: unpack_aos_to_soa_kernel<0>(recv_map, count, src, num_vars, fields); break;
    }
}

} // anonymous namespace

// ============================================================================
// --- Construction & Registration --------------------------------------------
// ============================================================================

HaloExchanger::HaloExchanger(const mesh::MeshPart& mp, const MPI_Comm comm)
    : comm_(comm), mp_(mp), nb_ranks_(mp.nb_ranks) {
    fields_phase_.tag = kFieldTag;
    grads_phase_.tag  = kGradLimiterTag;
}

void HaloExchanger::register_cell_fields(std::span<double* const> fields) {
    fields_phase_.ptrs.insert(fields_phase_.ptrs.end(), fields.begin(), fields.end());
    resize_phase(fields_phase_, "fields");
}

void HaloExchanger::register_grad_limiters(std::span<double* const> grad_bases,
                                           const std::size_t plane_stride,
                                           std::span<double* const> limiters) {
    // Flatten 3D gradient tensor planes into direct pointers:
    // Layout for this registered block: [dx vars][dy vars][dz vars][limiters]
    for (std::size_t plane = 0; plane < 3; ++plane) {
        const std::size_t offset = plane * plane_stride;
        for (double* base : grad_bases) {
            grads_phase_.ptrs.push_back(base + offset);
        }
    }
    grads_phase_.ptrs.insert(grads_phase_.ptrs.end(), limiters.begin(), limiters.end());
    resize_phase(grads_phase_, "gradients/limiters");
}

void HaloExchanger::resize_phase(PhaseData& phase, const char* phase_name) {
    if (nb_ranks_.empty()) {
        return;
    }

    const std::size_t vpc = phase.ptrs.size();
    const std::size_t total_send = mp_.send_offsets.empty() ? 0 : static_cast<std::size_t>(mp_.send_offsets.back());
    const std::size_t total_recv = mp_.recv_offsets.empty() ? 0 : static_cast<std::size_t>(mp_.recv_offsets.back());

    // Check MPI integer count overflow for both send and recv
    for (std::size_t k = 0; k + 1 < mp_.send_offsets.size(); ++k) {
        const std::int64_t count = static_cast<std::int64_t>(mp_.send_offsets[k + 1] - mp_.send_offsets[k]);
        if (count * static_cast<std::int64_t>(vpc) > kMpiCountLimit) {
            mpi::fatal(comm_, std::string("HaloExchanger: ") + phase_name + " send payload exceeds MPI int limit");
        }
    }

    for (std::size_t k = 0; k + 1 < mp_.recv_offsets.size(); ++k) {
        const std::int64_t count = static_cast<std::int64_t>(mp_.recv_offsets[k + 1] - mp_.recv_offsets[k]);
        if (count * static_cast<std::int64_t>(vpc) > kMpiCountLimit) {
            mpi::fatal(comm_, std::string("HaloExchanger: ") + phase_name + " recv payload exceeds MPI int limit");
        }
    }

    phase.send_buf.assign(total_send * vpc, 0.0);
    phase.recv_buf.assign(total_recv * vpc, 0.0);
    phase.requests.assign(2 * nb_ranks_.size(), MPI_REQUEST_NULL);
}

// ============================================================================
// --- Core MPI Communication Engine ------------------------------------------
// ============================================================================

void HaloExchanger::post_irecvs(PhaseData& phase) {
    const std::size_t nnb = nb_ranks_.size();
    const std::size_t vpc = phase.ptrs.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const std::size_t cell_offset = static_cast<std::size_t>(ro[k]);
        const std::size_t cell_count  = static_cast<std::size_t>(ro[k + 1] - ro[k]);
        const int mpi_count = static_cast<int>(cell_count * vpc);

        MPI_Irecv(phase.recv_buf.data() + cell_offset * vpc,
                  mpi_count,
                  MPI_DOUBLE,
                  nb_ranks_[k],
                  phase.tag,
                  comm_,
                  &phase.requests[k]);
    }
}

void HaloExchanger::pack_and_isend(PhaseData& phase) {
    const std::size_t nnb = nb_ranks_.size();
    const std::size_t vpc = phase.ptrs.size();
    const LocalIndex* CFD_RESTRICT so = mp_.send_offsets.data();
    const LocalIndex* CFD_RESTRICT sv = mp_.send_owned_local.data();
    double* const* fields = phase.ptrs.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const std::size_t cell_offset = static_cast<std::size_t>(so[k]);
        const std::size_t cell_count  = static_cast<std::size_t>(so[k + 1] - so[k]);
        const int mpi_count = static_cast<int>(cell_count * vpc);

        double* CFD_RESTRICT dst = phase.send_buf.data() + cell_offset * vpc;

        dispatch_pack(sv + cell_offset, cell_count, fields, vpc, dst);

        MPI_Isend(dst,
                  mpi_count,
                  MPI_DOUBLE,
                  nb_ranks_[k],
                  phase.tag,
                  comm_,
                  &phase.requests[nnb + k]);
    }
}

void HaloExchanger::unpack_ghosts(PhaseData& phase) {
    const std::size_t nnb = nb_ranks_.size();
    const std::size_t vpc = phase.ptrs.size();
    const LocalIndex* CFD_RESTRICT ro = mp_.recv_offsets.data();
    const LocalIndex* CFD_RESTRICT rv = mp_.recv_ghost_local.data();
    double* const* fields = phase.ptrs.data();

    for (std::size_t k = 0; k < nnb; ++k) {
        const std::size_t cell_offset = static_cast<std::size_t>(ro[k]);
        const std::size_t cell_count  = static_cast<std::size_t>(ro[k + 1] - ro[k]);
        const double* CFD_RESTRICT src = phase.recv_buf.data() + cell_offset * vpc;

        dispatch_unpack(rv + cell_offset, cell_count, src, vpc, fields);
    }
}

// ============================================================================
// --- Public Interface Dispatchers -------------------------------------------
// ============================================================================

void HaloExchanger::start_exchange_fields() {
    if (nb_ranks_.empty() || fields_phase_.ptrs.empty()) return;
    post_irecvs(fields_phase_);
    pack_and_isend(fields_phase_);
}

void HaloExchanger::finish_exchange_fields() {
    if (nb_ranks_.empty() || fields_phase_.ptrs.empty()) return;
    MPI_Waitall(static_cast<int>(fields_phase_.requests.size()),
                fields_phase_.requests.data(),
                MPI_STATUSES_IGNORE);
    unpack_ghosts(fields_phase_);
}

void HaloExchanger::exchange_fields() {
    start_exchange_fields();
    finish_exchange_fields();
}

void HaloExchanger::start_exchange_grad_limiters() {
    if (nb_ranks_.empty() || grads_phase_.ptrs.empty()) return;
    post_irecvs(grads_phase_);
    pack_and_isend(grads_phase_);
}

void HaloExchanger::finish_exchange_grad_limiters() {
    if (nb_ranks_.empty() || grads_phase_.ptrs.empty()) return;
    MPI_Waitall(static_cast<int>(grads_phase_.requests.size()),
                grads_phase_.requests.data(),
                MPI_STATUSES_IGNORE);
    unpack_ghosts(grads_phase_);
}

void HaloExchanger::exchange_grad_limiters() {
    start_exchange_grad_limiters();
    finish_exchange_grad_limiters();
}

} // namespace cfd::solver::halo
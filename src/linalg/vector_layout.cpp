#include "cfd/linalg/vector_layout.hpp"

#include <algorithm>
#include <cstring>

#include "cfd/linalg/types.hpp"

namespace cfd::linalg {

namespace {
// Message tag for ghost-value traffic. updateGhosts() fully completes before
// returning, so reusing one tag across calls is safe; the communicator is
// duplicated per layout, isolating the traffic from other library exchanges.
constexpr int kGhostExchangeTag = 4317;
}  // namespace

struct VectorLayout::Plan {
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

    // Scratch for updateGhosts (sized for the block size in use).
    mutable std::vector<double> send_buf, recv_buf;
    mutable std::vector<MPI_Request> requests;
    mutable int cached_bs = 0;

    ~Plan() {
        if (owns_comm && comm != MPI_COMM_SELF && comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }
};

VectorLayout::VectorLayout() : plan_(std::make_shared<Plan>()) {}

MPI_Comm VectorLayout::comm() const { return plan_->comm; }
int VectorLayout::rank() const { return plan_->rank; }
int VectorLayout::nprocs() const { return plan_->nprocs; }

GlobalIndex VectorLayout::globalSize() const { return plan_->n_global; }
LocalIndex VectorLayout::localSize() const { return plan_->n_local; }
GlobalIndex VectorLayout::localBegin() const { return plan_->begin; }

LocalIndex VectorLayout::ghostSize() const { return static_cast<LocalIndex>(plan_->ghosts.size()); }
const std::vector<GlobalIndex>& VectorLayout::ghostGlobalIds() const { return plan_->ghosts; }

VectorLayout::VectorLayout(MPI_Comm comm, GlobalIndex n_global, LocalIndex n_local)
    : plan_(std::make_shared<Plan>()) {
    check(n_global >= 0 && n_local >= 0, comm, "VectorLayout: sizes must be non-negative");
    check(static_cast<GlobalIndex>(n_local) <= n_global, comm,
          "VectorLayout: local rows exceed global rows");

    Plan& p = *plan_;
    p.n_global = n_global;
    p.n_local = n_local;

    if (comm != MPI_COMM_SELF && comm != MPI_COMM_NULL) {
        MPI_Comm_dup(comm, &p.comm);
        p.owns_comm = true;
        MPI_Comm_rank(p.comm, &p.rank);
        MPI_Comm_size(p.comm, &p.nprocs);
        GlobalIndex send = static_cast<GlobalIndex>(n_local);
        GlobalIndex prefix = 0;
        MPI_Exscan(&send, &prefix, 1, detail::mpi_index_type(), MPI_SUM, p.comm);
        p.begin = (p.rank == 0) ? 0 : prefix;
        GlobalIndex max_end = 0;
        const GlobalIndex end = p.begin + send;
        MPI_Allreduce(&end, &max_end, 1, detail::mpi_index_type(), MPI_MAX, p.comm);
        check(max_end == n_global, p.comm,
              "VectorLayout: sum of local rows does not match the global size");
    }
    // MPI_COMM_SELF path: rank 0 of size 1, begin = 0 — already the default.
}

bool VectorLayout::compatibleWith(const VectorLayout& other) const {
    if (plan_ == other.plan_) return true;
    const Plan& a = *plan_;
    const Plan& b = *other.plan_;
    int cmp = MPI_UNEQUAL;
    MPI_Comm_compare(a.comm, b.comm, &cmp);
    return (cmp == MPI_IDENT || cmp == MPI_CONGRUENT) && a.n_global == b.n_global &&
           a.begin == b.begin && a.n_local == b.n_local && a.ghosts == b.ghosts;
}

LocalIndex VectorLayout::localIndex(GlobalIndex global_row) const {
    const Plan& p = *plan_;
    if (global_row >= p.begin && global_row < p.begin + p.n_local) {
        return static_cast<LocalIndex>(global_row - p.begin);
    }
    const auto it = std::lower_bound(p.ghosts.begin(), p.ghosts.end(), global_row);
    if (it == p.ghosts.end() || *it != global_row) return kInvalidLocalIndex;
    return p.n_local + static_cast<LocalIndex>(it - p.ghosts.begin());
}

void VectorLayout::setGhosts(const std::vector<GlobalIndex>& ghost_ids) {
    Plan& p = *plan_;
    for (std::size_t i = 0; i < ghost_ids.size(); ++i) {
        check(ghost_ids[i] >= 0 && ghost_ids[i] < p.n_global, p.comm,
              "VectorLayout::setGhosts: ghost row outside the global range");
        check(ghost_ids[i] < p.begin || ghost_ids[i] >= p.begin + p.n_local, p.comm,
              "VectorLayout::setGhosts: owned row declared a ghost");
        if (i > 0) {
            check(ghost_ids[i] > ghost_ids[i - 1], p.comm,
                  "VectorLayout::setGhosts: ghost ids must be sorted and unique");
        }
    }
    p.ghosts = ghost_ids;

    p.all_begins.assign(static_cast<std::size_t>(p.nprocs), 0);
    MPI_Allgather(&p.begin, 1, detail::mpi_index_type(), p.all_begins.data(), 1,
                  detail::mpi_index_type(), p.comm);

    // Owner of a row under the contiguous distribution: ranks own disjoint
    // ascending ranges, so upper_bound gives the (last, non-empty) owner.
    auto owner_of = [&p](GlobalIndex g) {
        return static_cast<int>(
                   std::upper_bound(p.all_begins.begin(), p.all_begins.end(), g) -
                   p.all_begins.begin()) - 1;
    };

    // Ghost ids are sorted and owners are monotone in the id, so the requests
    // are already grouped by destination rank.
    std::vector<int> req_counts(static_cast<std::size_t>(p.nprocs), 0);
    for (GlobalIndex g : p.ghosts) {
        ++req_counts[static_cast<std::size_t>(owner_of(g))];
    }
    std::vector<int> ans_counts(static_cast<std::size_t>(p.nprocs), 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, ans_counts.data(), 1, MPI_INT, p.comm);

    std::vector<int> sdispls(static_cast<std::size_t>(p.nprocs), 0);
    std::vector<int> rdispls(static_cast<std::size_t>(p.nprocs), 0);
    int total_req = 0, total_ans = 0;
    for (int r = 0; r < p.nprocs; ++r) {
        const std::size_t ru = static_cast<std::size_t>(r);
        sdispls[ru] = total_req;
        total_req += req_counts[ru];
        rdispls[ru] = total_ans;
        total_ans += ans_counts[ru];
    }
    check(total_req >= 0 && static_cast<std::size_t>(total_req) == p.ghosts.size(), p.comm,
          "VectorLayout::setGhosts: internal request count mismatch");

    // Tell each owner which of its rows this rank ghosts.
    std::vector<GlobalIndex> requested(static_cast<std::size_t>(total_ans));
    MPI_Alltoallv(p.ghosts.data(), req_counts.data(), sdispls.data(), detail::mpi_index_type(),
                  requested.data(), ans_counts.data(), rdispls.data(), detail::mpi_index_type(),
                  p.comm);

    // My rows others ghost, grouped by requesting rank, in the request order.
    p.send_ranks.clear();
    p.send_counts.clear();
    p.send_displ.clear();
    p.send_idx.clear(); p.send_idx.reserve(requested.size());
    for (int q = 0; q < p.nprocs; ++q) {
        if (ans_counts[static_cast<std::size_t>(q)] == 0) continue;
        p.send_ranks.push_back(q);
        p.send_displ.push_back(static_cast<int>(p.send_idx.size()));
        p.send_counts.push_back(ans_counts[static_cast<std::size_t>(q)]);
        for (int e = 0; e < ans_counts[static_cast<std::size_t>(q)]; ++e) {
            const GlobalIndex g = requested[static_cast<std::size_t>(rdispls[static_cast<std::size_t>(q)] + e)];
            check(g >= p.begin && g < p.begin + p.n_local, p.comm,
                  "VectorLayout::setGhosts: inconsistent ownership, foreign row requested");
            p.send_idx.push_back(static_cast<LocalIndex>(g - p.begin));
        }
    }

    // Ghost slots I read, grouped by owning rank, in the order I sent them.
    p.recv_ranks.clear();
    p.recv_counts.clear();
    p.recv_displ.clear();
    p.recv_idx.clear(); p.recv_idx.reserve(p.ghosts.size());
    std::size_t gp = 0;
    for (int o = 0; o < p.nprocs; ++o) {
        if (req_counts[static_cast<std::size_t>(o)] == 0) continue;
        p.recv_ranks.push_back(o);
        p.recv_displ.push_back(static_cast<int>(p.recv_idx.size()));
        p.recv_counts.push_back(req_counts[static_cast<std::size_t>(o)]);
        for (int e = 0; e < req_counts[static_cast<std::size_t>(o)]; ++e) {
            p.recv_idx.push_back(p.n_local + static_cast<LocalIndex>(gp));
            ++gp;
        }
    }
    check(gp == p.ghosts.size(), p.comm, "VectorLayout::setGhosts: internal ghost slot mismatch");

    p.cached_bs = 0;  // force scratch resize on the next update
}

void VectorLayout::updateGhosts(double* values, int block_size) const {
    Plan& p = *plan_;
    const std::size_t ns = p.send_ranks.size();
    const std::size_t nr = p.recv_ranks.size();
    if (ns + nr == 0) return;
    check(block_size > 0, p.comm, "VectorLayout::updateGhosts: block size must be positive");

    if (p.cached_bs != block_size) {
        std::size_t tot_send = 0, tot_recv = 0;
        for (int c : p.send_counts) tot_send += static_cast<std::size_t>(c);
        for (int c : p.recv_counts) tot_recv += static_cast<std::size_t>(c);
        p.send_buf.resize(tot_send * static_cast<std::size_t>(block_size));
        p.recv_buf.resize(tot_recv * static_cast<std::size_t>(block_size));
        p.cached_bs = block_size;
    }

    // Pack owned rows for the neighbours.
    for (std::size_t s = 0; s < ns; ++s) {
        double* dst = p.send_buf.data() +
                      static_cast<std::size_t>(p.send_displ[s]) * static_cast<std::size_t>(block_size);
        for (int e = 0; e < p.send_counts[s]; ++e) {
            const std::size_t slot = static_cast<std::size_t>(p.send_idx[static_cast<std::size_t>(
                p.send_displ[s] + e)]);
            std::memcpy(dst + static_cast<std::size_t>(e) * static_cast<std::size_t>(block_size),
                        values + slot * static_cast<std::size_t>(block_size),
                        sizeof(double) * static_cast<std::size_t>(block_size));
        }
    }

    p.requests.resize(ns + nr);
    int rq = 0;
    for (std::size_t r = 0; r < nr; ++r) {
        MPI_Irecv(p.recv_buf.data() +
                      static_cast<std::size_t>(p.recv_displ[r]) * static_cast<std::size_t>(block_size),
                  p.recv_counts[r] * block_size, MPI_DOUBLE, p.recv_ranks[r], kGhostExchangeTag,
                  p.comm, &p.requests[static_cast<std::size_t>(rq++)]);
    }
    for (std::size_t s = 0; s < ns; ++s) {
        MPI_Isend(p.send_buf.data() +
                      static_cast<std::size_t>(p.send_displ[s]) * static_cast<std::size_t>(block_size),
                  p.send_counts[s] * block_size, MPI_DOUBLE, p.send_ranks[s], kGhostExchangeTag,
                  p.comm, &p.requests[static_cast<std::size_t>(rq++)]);
    }
    MPI_Waitall(static_cast<int>(ns + nr), p.requests.data(), MPI_STATUSES_IGNORE);

    // Scatter received values into the ghost slots.
    for (std::size_t r = 0; r < nr; ++r) {
        for (int e = 0; e < p.recv_counts[r]; ++e) {
            const std::size_t slot = static_cast<std::size_t>(p.recv_idx[static_cast<std::size_t>(
                p.recv_displ[r] + e)]);
            const double* src =
                p.recv_buf.data() +
                (static_cast<std::size_t>(p.recv_displ[r]) + static_cast<std::size_t>(e)) *
                    static_cast<std::size_t>(block_size);
            std::memcpy(values + slot * static_cast<std::size_t>(block_size), src,
                        sizeof(double) * static_cast<std::size_t>(block_size));
        }
    }
}

}  // namespace cfd::linalg

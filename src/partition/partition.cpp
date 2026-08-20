#include "cfd/partition/partition.hpp"

#include <dkaminpar.h>
#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "cfd/mesh/sfc.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

void partition_cells(RawMesh& m, DualGraph& g, PartitionResult& pr, int num_threads) {
    const int nprocs = m.nprocs;
    const long long nl = m.n_local();
    const long long n_cells = m.n_cells_g;

    // Trivial case: a single part on a single rank, nothing to partition.
    if (nprocs == 1) {
        pr.block.assign(nl, 0);
        pr.part2rank = {0};
        pr.edge_cut = 0;
        pr.imbalance = 1.0;
    }

    // --- KaMinPar (distributed, no central assembly) ---
    if (nprocs > 1) {  // see the trivial case above
        kaminpar::dist::Context ctx = kaminpar::dist::create_default_context();
        ctx.parallel.num_threads = static_cast<std::size_t>(num_threads);
        kaminpar::dKaMinPar part(MPI_COMM_WORLD, num_threads, ctx);
        part.set_output_level(kaminpar::OutputLevel::QUIET);
        if (g_verbose >= 2) part.set_output_level(kaminpar::OutputLevel::EXPERIMENT);

        std::vector<kaminpar::dist::GlobalNodeID> node_dist(m.cell_displ.begin(),
                                                            m.cell_displ.end());
        std::vector<kaminpar::dist::GlobalEdgeID> xadj(g.offsets.begin(), g.offsets.end());
        std::vector<kaminpar::dist::GlobalNodeID> adjncy(g.adj.begin(), g.adj.end());
        part.copy_graph(node_dist, xadj, adjncy);
        part.set_k(static_cast<kaminpar::dist::BlockID>(nprocs));
        part.set_uniform_max_block_weights(0.03);

        pr.block.resize(nl);
        const long long cut = part.compute_partition(std::span<kaminpar::dist::BlockID>(pr.block));
        pr.edge_cut = cut;
    }  // nprocs > 1

    // --- Balance ---
    std::vector<long long> per_part(nprocs, 0);
    for (uint32_t b : pr.block)
        if (b < static_cast<uint32_t>(nprocs)) ++per_part[b];
    MPI_Allreduce(MPI_IN_PLACE, per_part.data(), nprocs, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    const long long mx = *std::max_element(per_part.begin(), per_part.end());
    pr.imbalance = static_cast<double>(mx * nprocs) / static_cast<double>(n_cells);

    // --- Cell centroids (needed for the part centroids) ---
    std::vector<int32_t> used_nodes;
    std::vector<int> slot_of_node;  // slot (i*8+k) of every node in used_nodes
    used_nodes.reserve(nl * 8);
    slot_of_node.reserve(nl * 8);
    for (long long slot = 0; slot < nl * 8; ++slot)
        if (m.cnodes[slot] >= 0) {
            used_nodes.push_back(m.cnodes[slot]);
            slot_of_node.push_back(static_cast<int>(slot));
        }
    const std::vector<double> ncrd = fetch_coords(m, used_nodes);
    std::vector<double> slotcrd(nl * 8 * 3, 0.0);
    for (size_t j = 0; j < used_nodes.size(); ++j)
        for (int d = 0; d < 3; ++d) slotcrd[3 * slot_of_node[j] + d] = ncrd[3 * j + d];

    auto centroid = [&](long long i, double* c) {
        const int t = m.ctype[i];
        const int npt = kNodesPerType[t];
        c[0] = c[1] = c[2] = 0.0;
        for (int k = 0; k < npt; ++k) {
            const long long off = i * 8 + k;
            c[0] += slotcrd[3 * off + 0];
            c[1] += slotcrd[3 * off + 1];
            c[2] += slotcrd[3 * off + 2];
        }
        c[0] /= npt;
        c[1] /= npt;
        c[2] /= npt;
    };

    // --- Part centroids (allreduce over the k parts) ---
    std::vector<double> sums(nprocs * 4, 0.0);  // x,y,z,count
    {
        double c[3];
        for (long long i = 0; i < nl; ++i) {
            centroid(i, c);
            const uint32_t b = pr.block[i];
            sums[4 * b + 0] += c[0];
            sums[4 * b + 1] += c[1];
            sums[4 * b + 2] += c[2];
            sums[4 * b + 3] += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), nprocs * 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // --- Global node bbox for SFC normalization ---
    double lob[3] = {1e300, 1e300, 1e300}, hib[3] = {-1e300, -1e300, -1e300};
    for (long long i = 0; i < static_cast<long long>(m.my_node_coords.size()); i += 3)
        for (int d = 0; d < 3; ++d) {
            const double v = m.my_node_coords[i + d];
            lob[d] = std::min(lob[d], v);
            hib[d] = std::max(hib[d], v);
        }
    MPI_Allreduce(MPI_IN_PLACE, lob, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, hib, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // --- Parts -> ranks: SFC order of the part centroids ---
    std::vector<std::pair<uint64_t, int>> keyed(nprocs);
    for (int b = 0; b < nprocs; ++b) {
        const double cnt = std::max(1.0, sums[4 * b + 3]);
        double f[3];
        for (int d = 0; d < 3; ++d) {
            const double c = sums[4 * b + d] / cnt;
            f[d] = (hib[d] > lob[d]) ? (c - lob[d]) / (hib[d] - lob[d]) : 0.5;
        }
        keyed[b] = {sfc_key(f[0], f[1], f[2]), b};
    }
    std::sort(keyed.begin(), keyed.end());
    pr.part2rank.assign(nprocs, -1);
    for (int pos = 0; pos < nprocs; ++pos) pr.part2rank[keyed[pos].second] = pos;

    log_stat(
        "Partition: edge_cut=%lld (%.2f%% of edges), imbalance=%.3f, "
        "max_block=%lld",
        pr.edge_cut, n_cells > 0 ? 100.0 * pr.edge_cut / (3.0 * n_cells) : 0.0, pr.imbalance, mx);
    for (int b = 0; b < nprocs; ++b)
        log_stat("  part %d -> rank %d, cells %lld", b, pr.part2rank[b], per_part[b]);
}

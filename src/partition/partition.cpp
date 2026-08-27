#include "cfd/partition/partition.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include <dkaminpar.h>
#include <mpi.h>

#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace cfd::partition {

namespace {

// =============================================================================
// Cluster Topology Discovery
// =============================================================================

struct RankTopology {
    int node_id = -1;
    int local_rank = -1;
    int global_rank = -1;

    [[nodiscard]] bool operator<(const RankTopology& o) const noexcept {
        if (node_id != o.node_id) return node_id < o.node_id;
        return local_rank < o.local_rank;
    }
};

struct TopologyInfo {
    std::vector<int> part2rank;
    std::vector<int> rank_to_node;
};

// Discovers physical cluster compute nodes and constructs part->rank & rank->node tables
TopologyInfo build_topology_info(MPI_Comm comm, int nprocs, int rank) {
    const auto nprocs_sz = static_cast<std::size_t>(nprocs);

    // 1. Discover ranks sharing physical node memory
    MPI_Comm shared_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);

    int shared_rank = 0;
    MPI_Comm_rank(shared_comm, &shared_rank);

    // 2. Leader communicator among node master processes (shared_rank == 0)
    MPI_Comm leader_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm, shared_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leader_comm);

    int node_id = -1;
    int num_nodes = 0;
    if (shared_rank == 0) {
        MPI_Comm_rank(leader_comm, &node_id);
        MPI_Comm_size(leader_comm, &num_nodes);
    }

    MPI_Bcast(&node_id, 1, MPI_INT, 0, shared_comm);
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, shared_comm);

    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }
    MPI_Comm_free(&shared_comm);

    // 3. Gather full cluster topology to compute continuous block-to-node placement
    const RankTopology my_topo{node_id, shared_rank, rank};
    std::vector<RankTopology> all_topo(nprocs_sz);

    MPI_Allgather(&my_topo, static_cast<int>(sizeof(RankTopology)), MPI_BYTE,
                  all_topo.data(), static_cast<int>(sizeof(RankTopology)), MPI_BYTE, comm);

    // Build rank_to_node lookup
    std::vector<int> rank_to_node(nprocs_sz);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        rank_to_node[i] = all_topo[i].node_id;
    }

    // Sort ranks primarily by physical node, secondarily by core ID
    std::vector<RankTopology> sorted_topo = all_topo;
    std::sort(sorted_topo.begin(), sorted_topo.end());

    std::vector<int> part2rank(nprocs_sz);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        part2rank[i] = sorted_topo[i].global_rank;
    }

    mpi::log_stat("INFO[Partitioner]: Detected %d physical compute nodes across %d MPI ranks", num_nodes, nprocs);

    return TopologyInfo{std::move(part2rank), std::move(rank_to_node)};
}

// Binary search rank locator over monotonic (nprocs + 1) displacement array
inline int find_owner_rank(GlobalIndex gid, const std::vector<GlobalIndex>& displ) noexcept {
    auto it = std::upper_bound(displ.begin(), displ.end(), gid);
    return static_cast<int>(std::distance(displ.begin(), it) - 1);
}

// =============================================================================
// Ghost Target Rank Exchange for Cut Verification
// =============================================================================

struct alignas(8) GhostReqMsg {
    GlobalIndex cell_gid;  // Global ID of requested cell
    LocalIndex local_slot; // Local query index on requesting rank
};

struct alignas(8) GhostRespMsg {
    LocalIndex local_slot; // Local query index on requesting rank
    int target_rank;       // Resolved target rank
};

} // anonymous namespace

PartitionResult partition_cells(const mesh::RawMesh& m, const mesh::DualGraph& g, const DKaMinParOptions& dko) {
    // get local rank index and total ranks count
    const int nprocs = m.nprocs;
    const int rank = m.rank;
    const MPI_Comm comm = m.comm;

    // get local cell count
    const LocalIndex n_loc_cells = m.n_local_cells();

    // aux cast
    const auto n_loc_cells_sz = static_cast<std::size_t>(n_loc_cells);
    const auto nprocs_sz = static_cast<std::size_t>(nprocs);
    const PartitionBlockId num_blocks = (dko.block_count == 0) ? static_cast<PartitionBlockId>(nprocs) : dko.block_count;



    // -------------------------------------------------------------------------
    // Step 1: Discover Hardware Topology & Construct Part -> Rank Mapping
    // -------------------------------------------------------------------------
    PartitionResult result;
    const auto [part2rank, rank_to_node] = build_topology_info(comm, nprocs, rank);
    result.part2rank = part2rank;
    result.cell_target_rank.resize(n_loc_cells_sz);



    // -------------------------------------------------------------------------
    // Step 2: Prepare 64-bit Distributed CSR Data for dKaMinPar
    // -------------------------------------------------------------------------
    std::vector<kaminpar::dist::GlobalNodeID> vtxdist(nprocs_sz + 1);
    for (std::size_t i = 0; i <= nprocs_sz; ++i) {
        vtxdist[i] = static_cast<kaminpar::dist::GlobalNodeID>(m.cell_displ[i]);
    }

    std::vector<kaminpar::dist::GlobalEdgeID> xadj(n_loc_cells_sz + 1);
    for (std::size_t i = 0; i <= n_loc_cells_sz; ++i) {
        xadj[i] = static_cast<kaminpar::dist::GlobalEdgeID>(g.offsets[i]);
    }

    std::vector<kaminpar::dist::GlobalNodeID> adjncy(g.adj.size());
    for (std::size_t i = 0; i < g.adj.size(); ++i) {
        adjncy[i] = static_cast<kaminpar::dist::GlobalNodeID>(g.adj[i]);
    }



    // -------------------------------------------------------------------------
    // Step 3: Run dKaMinPar Graph Partitioning
    // -------------------------------------------------------------------------
    kaminpar::dKaMinPar::reseed(dko.seed);
    kaminpar::dKaMinPar partitioner(comm, dko.threads_per_rank, kaminpar::dist::create_default_context());

    partitioner.set_output_level(dko.quiet ? kaminpar::OutputLevel::QUIET : kaminpar::OutputLevel::EXPERIMENT);
    partitioner.copy_graph(vtxdist, xadj, adjncy);

    std::vector<kaminpar::dist::BlockID> raw_partition(n_loc_cells_sz, kaminpar::dist::kInvalidBlockID);

    const kaminpar::dist::GlobalEdgeWeight edge_cut_value = partitioner.compute_partition(
        static_cast<kaminpar::dist::BlockID>(num_blocks),
        dko.imbalance_tolerance,
        raw_partition
    );

    // Map raw partition block IDs to topology-aware target MPI ranks
    for (std::size_t c = 0; c < n_loc_cells_sz; ++c) {
        const std::size_t b = static_cast<std::size_t>(raw_partition[c]);
        result.cell_target_rank[c] = (b < result.part2rank.size()) ? result.part2rank[b] : static_cast<int>(b);
    }

    std::vector<PartitionBlockId> rank2part(static_cast<std::size_t>(nprocs));
    for (int block_id = 0; block_id < nprocs; ++block_id) {
        rank2part[static_cast<std::size_t>(part2rank[static_cast<std::size_t>(block_id)])] = static_cast<PartitionBlockId>(block_id);
    }
    result.rank2part = std::move(rank2part);



    // -------------------------------------------------------------------------
    // Step 4: Distributed Cut Verification & Inter-Node Metrics
    // -------------------------------------------------------------------------
    const GlobalIndex my_cell_start = m.cell_displ[static_cast<std::size_t>(rank)];
    const GlobalIndex my_cell_end = m.cell_displ[static_cast<std::size_t>(rank + 1)];

    std::vector<int> req_counts(nprocs_sz, 0);
    struct GhostQuery {
        GlobalIndex gid;
        LocalIndex edge_idx;
    };
    std::vector<GhostQuery> ghost_queries;

    // loop over my local cells
    for (LocalIndex u = 0; u < n_loc_cells; ++u) {
        const auto u_sz = static_cast<std::size_t>(u);
        const LocalIndex edge_start = g.offsets[u_sz];
        const LocalIndex edge_end = g.offsets[u_sz + 1];

        // loop over all my neighbors
        for (LocalIndex e = edge_start; e < edge_end; ++e) {
            // get neighbor global index
            const GlobalIndex v = g.adj[static_cast<std::size_t>(e)];

            // if neighbor belongs to other rank
            if (v < my_cell_start || v >= my_cell_end) {
                const int owner = find_owner_rank(v, m.cell_displ);
                ++req_counts[static_cast<std::size_t>(owner)];
                ghost_queries.push_back({v, e});
            }
        } // end loop over all my neighbors
    } // end loop over my local cells

    // Pack request messages
    std::vector<int> req_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        req_sdispls[i + 1] = req_sdispls[i] + req_counts[i];
    }

    std::vector<GhostReqMsg> req_send_buf(ghost_queries.size());
    std::vector<int> req_cursors = req_sdispls;

    for (std::size_t i = 0; i < ghost_queries.size(); ++i) {
        const auto& q = ghost_queries[i];
        const int owner = find_owner_rank(q.gid, m.cell_displ);
        req_send_buf[static_cast<std::size_t>(req_cursors[static_cast<std::size_t>(owner)]++)] 
            = GhostReqMsg{
            q.gid, static_cast<LocalIndex>(i)
        };
    }

    std::vector<GhostReqMsg> req_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, req_counts, req_send_buf, req_recv_buf);

    // Prepare response messages
    std::vector<GhostRespMsg> resp_send_buf(req_recv_buf.size());

    // loop over all requests
    for (std::size_t i = 0; i < req_recv_buf.size(); ++i) {
        const auto& req = req_recv_buf[i];
        const auto local_c = static_cast<LocalIndex>(req.cell_gid - my_cell_start);
        assert(local_c >= 0 && local_c < n_loc_cells);

        resp_send_buf[i] = GhostRespMsg{
            req.local_slot,
            result.cell_target_rank[static_cast<std::size_t>(local_c)]
        };
    } // end loop over all requests

    std::vector<int> resp_send_counts(nprocs_sz, 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, resp_send_counts.data(), 1, MPI_INT, comm);

    std::vector<GhostRespMsg> resp_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, resp_send_counts, resp_send_buf, resp_recv_buf);

    // Map ghost cell target ranks
    std::vector<int> ghost_target_ranks(ghost_queries.size(), -1);
    for (const auto& resp : resp_recv_buf) {
        ghost_target_ranks[static_cast<std::size_t>(resp.local_slot)] = resp.target_rank;
    }

    // Count local edge cuts
    GlobalIndex local_global_cut = 0;
    GlobalIndex local_node_cut = 0;
    std::size_t ghost_cursor = 0;

    // loop over all local cells
    for (LocalIndex u = 0; u < n_loc_cells; ++u) {
        const std::size_t u_sz = static_cast<std::size_t>(u);
        const GlobalIndex u_gid = m.global_cell_id(u);
        const int target_u = result.cell_target_rank[u_sz];
        const int node_u = rank_to_node[static_cast<std::size_t>(target_u)];

        const LocalIndex edge_start = g.offsets[u_sz];
        const LocalIndex edge_end = g.offsets[u_sz + 1];

        // loop over all cell neighbors
        for (LocalIndex e = edge_start; e < edge_end; ++e) {
            const GlobalIndex v_gid = g.adj[static_cast<std::size_t>(e)];

            int target_v = -1;
            if (v_gid >= my_cell_start && v_gid < my_cell_end) {
                target_v = result.cell_target_rank[static_cast<std::size_t>(v_gid - my_cell_start)];
            } else {
                target_v = ghost_target_ranks[ghost_cursor++];
            }

            if (u_gid < v_gid) {
                if (target_u != target_v) {
                    ++local_global_cut;
                    const int node_v = rank_to_node[static_cast<std::size_t>(target_v)];
                    if (node_u != node_v) {
                        ++local_node_cut;
                    }
                }
            }
        } // end loop over all cell neighbors
    }

    const GlobalIndex cuts[2] = {local_global_cut, local_node_cut};
    GlobalIndex total_cuts[2] = {0, 0};
    MPI_Allreduce(cuts, total_cuts, 2, MPI_INT64_T, MPI_SUM, comm);

    result.global_edge_cut = total_cuts[0];
    result.inter_node_cut = total_cuts[1];

    if (result.global_edge_cut != edge_cut_value) {
        mpi::fatal(comm, "edge_cut_value dont consitstent with result.global_edge_cut");
    }

    mpi::log_stat(
            "INFO[Partitioner]: Partitioning complete. Global edge-cut = %lld, Inter-node network cut = %lld (%.1f%% of total cut)",
            static_cast<long long>(result.global_edge_cut),
            static_cast<long long>(result.inter_node_cut),
            result.global_edge_cut > 0 ? (100.0 * static_cast<double>(result.inter_node_cut) / static_cast<double>(result.global_edge_cut)) : 0.0
        );

    // count number of cells per each block
    std::vector<PartitionBlockId> local_ncells_per_rank(static_cast<std::size_t>(nprocs), 0);
    for (std::size_t i = 0; i < n_loc_cells_sz; ++i) {
        const int target = result.cell_target_rank[i];
        ++local_ncells_per_rank[static_cast<std::size_t>(target)];
    }

    std::vector<PartitionBlockId> global_ncells_per_rank(static_cast<std::size_t>(nprocs), 0);
    MPI_Allreduce(
        local_ncells_per_rank.data(), 
        global_ncells_per_rank.data(), 
        nprocs, 
        MPI_UINT32_T, 
        MPI_SUM, 
        comm
    );

    if (rank == 0) {
        for (int rank_id = 0; rank_id < nprocs; ++rank_id) {
            mpi::log_stat("[r%d] INFO[Partitioner], Number of cells = %d", rank_id, global_ncells_per_rank[static_cast<std::size_t>(rank_id)]);
        }
    }
                  
    return result;
}

} // namespace cfd::partition
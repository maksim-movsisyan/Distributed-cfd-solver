#include "cfd/mesh/localmesh.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

#include "cfd/mesh/cgns_reader.hpp"
#include "cfd/mesh/faces.hpp"
#include "cfd/mesh/geometry.hpp"
#include "cfd/mesh/sfc.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"
#include "cfd/partition/partition.hpp"

namespace {

// Cell payload in messages (migration / ghost delivery).
struct CellMig {
    int32_t gid;
    uint8_t type;
    int32_t nodes[8];
};
static_assert(sizeof(CellMig) == 40);

// Request to a node owner: cell (of rank crank) uses node.
struct SibReq {
    int32_t node, cell, crank;
};
// Reply: cell has a vertex-sharing neighbour sib (of rank srank).
struct SibResp {
    int32_t cell, sib, srank;
};

}  // namespace

void build_local_mesh(RawMesh& m, std::vector<FaceRec>& faces, const PartitionResult& pr,
                      MeshPart& mp) {
    const int nprocs = m.nprocs, rank = m.rank;
    const long long nl = m.n_local();

    mp.rank = rank;
    mp.nprocs = nprocs;
    mp.n_cells_g = m.n_cells_g;
    mp.n_nodes_g = m.n_nodes_g;
    mp.patches.clear();
    for (auto& pl : m.patch_list) mp.patches.push_back({pl.first, pl.second});

    // ============= 1. Migrate cells and faces to their final ranks ===========
    // After this step every rank stores exactly the cells whose final rank
    // is this rank (not yet sorted), plus every face whose smaller cell
    // landed here.
    std::vector<std::vector<CellMig>> cell_out(nprocs);
    std::vector<std::vector<FaceRec>> face_out(nprocs);
    std::vector<int32_t> target(nl);
    for (long long i = 0; i < nl; ++i) {
        target[i] = pr.part2rank[pr.block[i]];
        CellMig cm{};
        cm.gid = m.cgid[i];
        cm.type = m.ctype[i];
        for (int k = 0; k < 8; ++k) cm.nodes[k] = m.cnodes[i * 8 + k];
        cell_out[target[i]].push_back(cm);
    }
    for (const FaceRec& f : faces) face_out[target[f.cell_a - m.cell_displ[rank]]].push_back(f);

    auto cell_in = alltoall_vec(cell_out);
    cell_out.clear();
    auto face_in = alltoall_vec(face_out);
    face_out.clear();
    faces.clear();
    for (auto& buf : face_in) faces.insert(faces.end(), buf.begin(), buf.end());
    face_in.clear();

    // Owned cells: sort by gid (a canonical deterministic state).
    std::vector<CellMig> cells;
    for (auto& buf : cell_in) cells.insert(cells.end(), buf.begin(), buf.end());
    cell_in.clear();
    std::sort(cells.begin(), cells.end(),
              [](const CellMig& a, const CellMig& b) { return a.gid < b.gid; });
    const int n_own = static_cast<int>(cells.size());
    mp.n_own = n_own;

    std::unordered_map<int32_t, int32_t> gid2own;
    gid2own.reserve(n_own * 2);
    for (int i = 0; i < n_own; ++i) gid2own[cells[i].gid] = i;

    // ===== 2. Ghost layer: vertex-sharing neighbours (covers face ones) =======
    // Face neighbours are a subset of vertex-sharing ones, so a single round
    // yields the full one-hop ghost layer (required for LS gradients).
    // After this step every rank knows the complete set of ghost cells it
    // needs: foreign cells sharing at least one node with its owned cells.
    // One round produces the whole 1-hop layer (needed for LS gradients).
    std::unordered_map<int32_t, int32_t> ghost_donor;  // gid -> donor rank
    if (nprocs > 1) {
        std::vector<std::vector<SibReq>> req(nprocs);
        for (int i = 0; i < n_own; ++i) {
            const int t = cells[i].type;
            const int npt = kNodesPerType[t];
            for (int k = 0; k < npt; ++k)
                req[block_owner(cells[i].nodes[k], m.node_displ)].push_back(
                    {cells[i].nodes[k], cells[i].gid, static_cast<int32_t>(rank)});
        }
        auto at_owner = alltoall_vec(req);
        req.clear();

        // Node owner: full per-node cell lists -> replies to every participant.
        std::vector<std::vector<SibResp>> resp(nprocs);
        {
            std::unordered_map<int32_t, std::vector<std::pair<int32_t, int32_t>>> node2cells;
            for (int p = 0; p < nprocs; ++p)
                for (const SibReq& r : at_owner[p]) node2cells[r.node].push_back({r.cell, r.crank});
            for (const auto& kv : node2cells) {
                const auto& lst = kv.second;
                for (const auto& me : lst)
                    for (const auto& other : lst)
                        if (other != me)
                            resp[me.second].push_back({me.first, other.first, other.second});
            }
        }
        at_owner.clear();
        auto back = alltoall_vec(resp);
        resp.clear();
        // Ghosts = foreign cells absent from my owned set.
        for (auto& buf : back)
            for (const SibResp& r : buf)
                if (r.srank != rank && !gid2own.count(r.sib)) ghost_donor[r.sib] = r.srank;
        back.clear();
    }
    mp.n_cells = n_own + static_cast<int>(ghost_donor.size());

    // ==================== 3. SFC ordering of owned cells =====================
    {
        std::vector<int32_t> used;
        used.reserve(static_cast<size_t>(n_own) * 8);
        std::vector<int> slot_of;
        slot_of.reserve(static_cast<size_t>(n_own) * 8);
        for (int i = 0; i < n_own; ++i)
            for (int k = 0; k < 8; ++k)
                if (cells[i].nodes[k] >= 0) {
                    used.push_back(cells[i].nodes[k]);
                    slot_of.push_back(i * 8 + k);
                }
        const std::vector<double> crd = fetch_coords(m, used);
        std::vector<double> scrd(n_own * 8 * 3, 0.0);
        for (size_t j = 0; j < used.size(); ++j)
            for (int d = 0; d < 3; ++d) scrd[3 * slot_of[j] + d] = crd[3 * j + d];

        // Global bbox (for SFC normalization) from this rank's nodes.
        double lob[3] = {1e300, 1e300, 1e300}, hib[3] = {-1e300, -1e300, -1e300};
        for (size_t i = 0; i < m.my_node_coords.size(); i += 3)
            for (int d = 0; d < 3; ++d) {
                lob[d] = std::min(lob[d], m.my_node_coords[i + d]);
                hib[d] = std::max(hib[d], m.my_node_coords[i + d]);
            }
        MPI_Allreduce(MPI_IN_PLACE, lob, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hib, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        for (int d = 0; d < 3; ++d) {
            mp.bbox_lo[d] = lob[d];
            mp.bbox_hi[d] = hib[d];
        }

        std::vector<std::pair<uint64_t, int>> keyed(n_own);
        for (int i = 0; i < n_own; ++i) {
            const int npt = kNodesPerType[cells[i].type];
            double c[3] = {0, 0, 0};
            for (int k = 0; k < npt; ++k)
                for (int d = 0; d < 3; ++d) c[d] += scrd[3 * (i * 8 + k) + d];
            for (int d = 0; d < 3; ++d) c[d] /= npt;
            double f[3];
            for (int d = 0; d < 3; ++d)
                f[d] = (hib[d] > lob[d]) ? (c[d] - lob[d]) / (hib[d] - lob[d]) : 0.5;
            keyed[i] = {sfc_key(f[0], f[1], f[2]), i};
        }
        std::sort(keyed.begin(), keyed.end());
        std::vector<CellMig> sorted_cells(n_own);
        for (int i = 0; i < n_own; ++i) sorted_cells[i] = cells[keyed[i].second];
        cells = std::move(sorted_cells);
        gid2own.clear();
        for (int i = 0; i < n_own; ++i) gid2own[cells[i].gid] = i;
    }

    // ================ 4. Fetch the ghost cells from their donors ==============
    // Order (donor, gid): it defines the canonical order of the recv maps.
    std::vector<std::pair<int32_t, int32_t>> ghosts(ghost_donor.begin(), ghost_donor.end());
    ghost_donor.clear();
    std::sort(ghosts.begin(), ghosts.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second < b.second : a.first < b.first;
    });
    // Send maps: which of my owned cells the neighbours need (the donor fills
    // them from incoming requests, preserving the requester's order).
    std::vector<std::vector<CellMig>> ghost_resp(nprocs);
    {
        std::vector<std::vector<int32_t>> greq(nprocs);
        for (size_t i = 0; i < ghosts.size(); ++i)
            greq[ghosts[i].second].push_back(ghosts[i].first);
        auto gin = alltoall_vec(greq);
        greq.clear();
        // Replies + send maps remembered in the request order.
        mp.send_offsets.assign(1, 0);
        for (int p = 0; p < nprocs; ++p) {
            for (int32_t g : gin[p]) {
                auto it = gid2own.find(g);
                if (it == gid2own.end()) {
                    log_rank(
                        "ghost request for a foreign cell %d: an internal "
                        "error",
                        g);
                    MPI_Abort(MPI_COMM_WORLD, 3);
                }
                mp.send_owned_local.push_back(it->second);
                ghost_resp[p].push_back(cells[it->second]);
            }
            mp.send_offsets.push_back(static_cast<int>(mp.send_owned_local.size()));
        }
    }
    auto gback = alltoall_vec(ghost_resp);
    ghost_resp.clear();
    {
        size_t i = 0;
        for (int p = 0; p < nprocs; ++p)
            for (const CellMig& c : gback[p]) {
                (void)c;
                ++i;
            }
        // The received count must match the number of requested ghosts.
        if (i != ghosts.size()) {
            log_rank("ghost exchange: received %zu, expected %zu", i, ghosts.size());
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    // Assemble the cell arrays: owned + ghosts.
    mp.cell_type.resize(mp.n_cells);
    mp.cell_gid.resize(mp.n_cells);
    mp.cell_donor.assign(mp.n_cells, -1);
    mp.cell_nodes.assign(static_cast<size_t>(mp.n_cells) * 8, -1);
    for (int i = 0; i < n_own; ++i) {
        mp.cell_type[i] = cells[i].type;
        mp.cell_gid[i] = cells[i].gid;
        for (int k = 0; k < 8; ++k) mp.cell_nodes[i * 8 + k] = cells[i].nodes[k];
    }
    {
        int i = n_own;
        for (int p = 0; p < nprocs; ++p)
            for (const CellMig& c : gback[p]) {
                mp.cell_type[i] = c.type;
                mp.cell_gid[i] = c.gid;
                mp.cell_donor[i] = p;
                for (int k = 0; k < 8; ++k) mp.cell_nodes[i * 8 + k] = c.nodes[k];
                ++i;
            }
    }
    gback.clear();
    std::unordered_map<int32_t, int32_t> gid2local;
    gid2local.reserve(mp.n_cells * 2);
    for (int i = 0; i < mp.n_cells; ++i) gid2local[mp.cell_gid[i]] = i;

    // ============== 5. Nodes: compact local numbering =========================
    // After this step every rank has its own compact node numbering: nodes
    // used by owned cells first (own nodes), then the remaining nodes
    // referenced by ghosts and faces; cell connectivity is already local.
    std::vector<int32_t> owned_nodes, ghost_nodes;
    {
        std::vector<int32_t> all;
        all.reserve(static_cast<size_t>(mp.n_cells) * 8);
        for (int i = 0; i < mp.n_cells; ++i)
            for (int k = 0; k < 8; ++k)
                if (mp.cell_nodes[i * 8 + k] >= 0) all.push_back(mp.cell_nodes[i * 8 + k]);
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        // owned nodes: used by owned cells
        std::vector<int32_t> ownset;
        ownset.reserve(static_cast<size_t>(n_own) * 8);
        for (int i = 0; i < n_own; ++i)
            for (int k = 0; k < 8; ++k)
                if (mp.cell_nodes[i * 8 + k] >= 0) ownset.push_back(mp.cell_nodes[i * 8 + k]);
        std::sort(ownset.begin(), ownset.end());
        ownset.erase(std::unique(ownset.begin(), ownset.end()), ownset.end());
        std::unordered_set<int32_t> ownhash(ownset.begin(), ownset.end());
        for (int32_t g : all)
            if (ownhash.count(g))
                owned_nodes.push_back(g);
            else
                ghost_nodes.push_back(g);
    }
    mp.n_nodes_own = static_cast<int>(owned_nodes.size());
    mp.n_nodes = mp.n_nodes_own + static_cast<int>(ghost_nodes.size());
    mp.node_gid = owned_nodes;
    mp.node_gid.insert(mp.node_gid.end(), ghost_nodes.begin(), ghost_nodes.end());
    std::unordered_map<int32_t, int32_t> node2local;
    node2local.reserve(mp.n_nodes * 2);
    for (int i = 0; i < mp.n_nodes; ++i) node2local[mp.node_gid[i]] = i;
    for (auto& v : mp.cell_nodes)
        if (v >= 0) v = node2local[v];
    // Coordinates of all local nodes.
    {
        const std::vector<double> crd = fetch_coords(m, mp.node_gid);
        mp.node_xyz = crd;
    }

    // ================= 6. Communication maps ==================================
    {
        // recv: ghost indices grouped by neighbour (the (donor, gid) order is fixed).
        mp.recv_offsets.assign(1, 0);
        int i = n_own;
        while (i < mp.n_cells) {
            const int32_t donor = mp.cell_donor[i];
            int j = i;
            while (j < mp.n_cells && mp.cell_donor[j] == donor) ++j;
            mp.nb_ranks.push_back(donor);
            for (int q = i; q < j; ++q) mp.recv_ghost_local.push_back(q);
            mp.recv_offsets.push_back(static_cast<int>(mp.recv_ghost_local.size()));
            i = j;
        }
        // send maps: step 4 accumulated segments over ALL ranks (nprocs+1);
        // switch to neighbour indexing (nnb+1). The order of send_owned_local
        // (ascending rank p) matches the sorted nb_ranks.
        const std::vector<int> send_by_rank = mp.send_offsets;  // nprocs+1
        mp.send_offsets.assign(1, 0);
        for (int n = 0; n < mp.n_neighbors(); ++n) {
            const int q = mp.nb_ranks[n];
            mp.send_offsets.push_back(send_by_rank[q + 1]);
        }
    }

    // =============== 6.5 Cell orientation check ================================
    // Volume < 0 -> permute the nodes via kOrientationFlip. Done BEFORE the
    // face-node capture so that the face node order matches the fixed
    // connectivity.
    {
        long long flipped = 0;
        for (int i = 0; i < mp.n_cells; ++i) {
            const CellType t = static_cast<CellType>(mp.cell_type[i]);
            const int npt = kNodesPerType[static_cast<int>(t)];
            auto vol = [&]() {
                std::vector<double> pts(3 * npt);
                for (int k = 0; k < npt; ++k) {
                    const int32_t n = mp.cell_nodes[i * 8 + k];
                    for (int d = 0; d < 3; ++d) pts[3 * k + d] = mp.node_xyz[3 * n + d];
                }
                return poly_cell_volume(t, pts.data());
            };
            if (vol() < 0.0) {
                int32_t oldn[8];
                for (int k = 0; k < 8; ++k) oldn[k] = mp.cell_nodes[i * 8 + k];
                for (int k = 0; k < 8; ++k)
                    mp.cell_nodes[i * 8 + k] = oldn[kOrientationFlip[static_cast<int>(t)][k]];
                if (vol() <= 0.0) {
                    log_rank("the permutation failed to fix orientation of cell gid %d",
                             mp.cell_gid[i]);
                    MPI_Abort(MPI_COMM_WORLD, 3);
                }
                ++flipped;
            }
        }
        mp.n_flipped = ll_sum(flipped);
        if (mp.n_flipped > 0)
            log_info(
                "Orientation: fixed %lld cells with a negative volume "
                "(of %lld)",
                mp.n_flipped, mp.n_cells_g);
    }

    // ================= 7. Faces ===============================================
    // (geometry and patch ids are filled in geometry.cpp / bc.cpp)
    const int nf = static_cast<int>(faces.size());
    mp.n_faces = nf;
    mp.face_owner.resize(nf);
    mp.face_neigh.assign(nf, -1);
    mp.face_type.resize(nf);
    mp.face_nodes.assign(static_cast<size_t>(nf) * 4, -1);
    mp.face_patch.assign(nf, -1);
    mp.face_donor.assign(nf, -1);
    {
        // Key -> (position in faces) for BC matching.
        std::vector<int> idx(nf);
        for (int i = 0; i < nf; ++i) idx[i] = i;
        for (int i = 0; i < nf; ++i) {
            const FaceRec& f = faces[i];
            const auto oa = gid2own.find(f.cell_a);
            if (oa == gid2own.end()) {
                log_rank("face without a local owner cell (gid %d)", f.cell_a);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            mp.face_owner[i] = oa->second;
            if (f.cell_b >= 0) {
                const auto nb = gid2local.find(f.cell_b);
                if (nb == gid2local.end()) {
                    log_rank("face neighbour %d not found in the local+ghost set", f.cell_b);
                    MPI_Abort(MPI_COMM_WORLD, 3);
                }
                mp.face_neigh[i] = nb->second;
                if (mp.cell_donor[nb->second] >= 0) mp.face_donor[i] = mp.cell_donor[nb->second];
            }
            // Face nodes: the canonical order from the owner-cell tables. The FaceRec
            // key holds global node ids while cell_nodes are already local, so compare
            // through the node_gid back-map.
            const int t = mp.cell_type[oa->second];
            const int32_t* cn = &mp.cell_nodes[oa->second * 8];
            bool found = false;
            for (int ff = 0; ff < kFacesPerType[t] && !found; ++ff) {
                const int nn = kFaceNodes[t][ff];
                int32_t tmp[4] = {-1, -1, -1, -1};
                for (int j = 0; j < nn; ++j) tmp[j] = cn[kFaceTable[t][ff][j]];
                int32_t srt[4] = {-1, -1, -1, -1};
                for (int j = 0; j < nn; ++j) srt[j] = mp.node_gid[tmp[j]];
                std::sort(srt, srt + 4);
                if (std::equal(srt, srt + 4, f.key.v)) {
                    for (int j = 0; j < nn; ++j) mp.face_nodes[i * 4 + j] = tmp[j];
                    mp.face_type[i] =
                        static_cast<uint8_t>(nn == 3 ? CellType::TRI : CellType::QUAD);
                    found = true;
                }
            }
            if (!found) {
                log_rank(
                    "no canonical face found for the key of cell %d "
                    "(cell_b=%d)",
                    f.cell_a, f.cell_b);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
        }
        // Sort by (owner, neigh) for cache locality in the face loop.
        std::vector<int> order(nf);
        for (int i = 0; i < nf; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (mp.face_owner[a] != mp.face_owner[b]) return mp.face_owner[a] < mp.face_owner[b];
            return mp.face_neigh[a] < mp.face_neigh[b];
        });
#define PERMUTE_SCALAR(vec)                                  \
    do {                                                     \
        auto tmp = vec;                                      \
        for (int i = 0; i < nf; ++i) tmp[i] = vec[order[i]]; \
        vec = std::move(tmp);                                \
    } while (0)
        PERMUTE_SCALAR(mp.face_owner);
        PERMUTE_SCALAR(mp.face_neigh);
        PERMUTE_SCALAR(mp.face_type);
        PERMUTE_SCALAR(mp.face_patch);
        PERMUTE_SCALAR(mp.face_donor);
        {
            std::vector<int32_t> tmp(static_cast<size_t>(nf) * 4, -1);
            for (int i = 0; i < nf; ++i)
                for (int j = 0; j < 4; ++j) tmp[i * 4 + j] = mp.face_nodes[order[i] * 4 + j];
            mp.face_nodes = std::move(tmp);
        }
        // Reorder faces (FaceRec) too: bc.cpp relies on the same order.
        {
            std::vector<FaceRec> tmp(nf);
            for (int i = 0; i < nf; ++i) tmp[i] = faces[order[i]];
            faces = std::move(tmp);
        }
    }

    // ================= 8. Global statistics ===================================
    mp.n_faces_g = ll_sum(nf);
    long long nbown = 0;
    for (int i = 0; i < nf; ++i) nbown += (mp.face_neigh[i] < 0);
    mp.n_bfaces_g = ll_sum(nbown);
}

// ----------------------------------------------------------------------------
bool meshpart_sane(const MeshPart& mp, std::string& err) {
    auto fail = [&](const char* e) {
        err = e;
        return false;
    };
    if (mp.n_own < 0 || mp.n_cells < mp.n_own) fail("n_own/n_cells invariants");
    if ((int)mp.cell_gid.size() != mp.n_cells) fail("cell_gid size");
    if ((int)mp.cell_nodes.size() != 8 * mp.n_cells) fail("cell_nodes size");
    for (int i = 0; i < mp.n_own; ++i)
        if (mp.cell_donor[i] != -1) fail("owned cell with a donor");
    for (int i = mp.n_own; i < mp.n_cells; ++i)
        if (mp.cell_donor[i] < 0) fail("ghost cell without a donor");
    for (int i = 0; i < mp.n_faces; ++i) {
        if (mp.face_owner[i] < 0 || mp.face_owner[i] >= mp.n_own) fail("face_owner outside owned");
        if (mp.face_neigh[i] >= mp.n_cells) fail("face_neigh outside cells");
    }
    return true;
}

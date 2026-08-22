#include "cfd/mesh/faces.hpp"

#include <mpi.h>

#include <algorithm>
#include <unordered_map>

#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace {

// CSR assembly message: cell self has neighbour other.
struct EdgeMsg {
    int32_t self, other;
};

}  // namespace

void build_faces(RawMesh& m, std::vector<FaceRec>& faces, DualGraph& g, FaceStats& st) {
    const int nprocs = m.nprocs, rank = m.rank;
    const long long nl = m.n_local();

    // --- 1. Generate the faces of own cells ---
    std::vector<std::vector<FaceRec>> outgoing(nprocs);
    {
        long long reserve_hint = 4 * nl / nprocs + 16;
        for (int p = 0; p < nprocs; ++p) outgoing[p].reserve(reserve_hint);
    }
    for (long long i = 0; i < nl; ++i) {
        const int t = m.ctype[i];
        const int nf = kFacesPerType[t];
        const int32_t* cn = &m.cnodes[i * 8];
        for (int f = 0; f < nf; ++f) {
            const int nn = kFaceNodes[t][f];
            FaceRec fr;
            for (int j = 0; j < nn; ++j) fr.key.v[j] = cn[kFaceTable[t][f][j]];
            std::sort(fr.key.v, fr.key.v + 4);  // the -1 tail stays at the end
            fr.cell_a = m.cgid[i];
            fr.cell_b = -1;
            outgoing[block_owner(fr.key.v[0], m.node_displ)].push_back(fr);
        }
    }

    // --- 2. Deduplicate on the minimal-node owners ---
    auto dedup = alltoall_vec(outgoing);
    outgoing.clear();
    std::unordered_map<FaceKey, FaceRec, FaceKeyHash> table;
    table.reserve(dedup.size() * 2 + 16);
    long long nonmanifold = 0;
    for (auto& buf : dedup) {
        for (FaceRec& fr : buf) {
            auto it = table.find(fr.key);
            if (it == table.end()) {
                table.emplace(fr.key, fr);
            } else if (it->second.cell_b < 0) {
                // second cell; the smaller gid becomes cell_a
                if (fr.cell_a < it->second.cell_a) {
                    it->second.cell_b = it->second.cell_a;
                    it->second.cell_a = fr.cell_a;
                } else {
                    it->second.cell_b = fr.cell_a;
                }
            } else {
                ++nonmanifold;  // a third cell at one face
            }
        }
    }
    dedup.clear();
    nonmanifold = ll_sum(nonmanifold);
    if (nonmanifold > 0) {
        log_rank("NON-MANIFOLD MESH: %lld faces with 3+ cells, aborting", nonmanifold);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // --- 3. Face -> owner of the smaller cell; edges -> both sides ---
    std::vector<std::vector<FaceRec>> face_out(nprocs);
    std::vector<std::vector<EdgeMsg>> edge_out(nprocs);
    for (const auto& kv : table) {
        const FaceRec& fr = kv.second;
        const int pa = block_owner(fr.cell_a, m.cell_displ);
        face_out[pa].push_back(fr);
        if (fr.cell_b >= 0) {  // dual-graph edge: interior faces only
            edge_out[pa].push_back({fr.cell_a, fr.cell_b});
            const int pb = block_owner(fr.cell_b, m.cell_displ);
            edge_out[pb].push_back({fr.cell_b, fr.cell_a});
        }
    }
    table.clear();

    auto face_in = alltoall_vec(face_out);
    face_out.clear();
    auto edge_in = alltoall_vec(edge_out);
    edge_out.clear();

    faces.clear();
    for (auto& buf : face_in) faces.insert(faces.end(), buf.begin(), buf.end());
    face_in.clear();

    // --- 4. CSR dual graph ---
    g.offsets.assign(nl + 1, 0);
    std::vector<int32_t> cnt(nl + 1, 0);
    for (auto& buf : edge_in)
        for (const EdgeMsg& e : buf) ++cnt[e.self - m.cell_displ[rank]];
    for (long long i = 0; i < nl; ++i) g.offsets[i + 1] = g.offsets[i] + cnt[i];
    g.adj.assign(g.offsets[nl], -1);
    {
        std::vector<long long> pos(g.offsets.begin(), g.offsets.end() - 1);
        for (auto& buf : edge_in)
            for (const EdgeMsg& e : buf) g.adj[pos[e.self - m.cell_displ[rank]]++] = e.other;
        edge_in.clear();
    }

    // --- Statistics ---
    long long nb = 0;
    for (const FaceRec& f : faces) nb += (f.cell_b < 0);
    st.n_faces_g = ll_sum(faces.size());
    st.n_bfaces_g = ll_sum(nb);
    st.n_ifaces_g = st.n_faces_g - st.n_bfaces_g;
    // Global counts only: per-rank record numbers carry no information.
    log_stat("Faces: global %lld (boundary %lld, interior %lld)", st.n_faces_g, st.n_bfaces_g,
             st.n_ifaces_g);

    // Consistency check: sum(cell faces) = 2*interior + boundary.
    long long sum_cell_faces = 0;
    for (long long i = 0; i < nl; ++i) sum_cell_faces += kFacesPerType[m.ctype[i]];
    const long long scf = ll_sum(sum_cell_faces);
    const long long expect_b = scf - 2 * st.n_ifaces_g;
    if (expect_b != st.n_bfaces_g) {
        log_rank("FACE INCONSISTENCY: expected %lld boundary faces, got %lld", expect_b,
                 st.n_bfaces_g);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

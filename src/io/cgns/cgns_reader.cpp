#include "cfd/mesh/cgns_reader.hpp"

#include <cgnslib.h>
#include <mpi.h>
#include <pcgnslib.h>

#include <algorithm>
#include <cstdio>
#include <unordered_map>

#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace {

[[noreturn]] void fatal(const std::string& what) {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    std::fprintf(stderr, "[r%d] CGNS reader error: %s\n", r, what.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

CellType cgns_elem_to_type(CGNS_ENUMT(ElementType_t) e) {
    switch (e) {
        case CGNS_ENUMV(TETRA_4):
            return CellType::TET;
        case CGNS_ENUMV(PYRA_5):
            return CellType::PYRA;
        case CGNS_ENUMV(PENTA_6):
            return CellType::PRISM;
        case CGNS_ENUMV(HEXA_8):
            return CellType::HEXA;
        case CGNS_ENUMV(TRI_3):
            return CellType::TRI;
        case CGNS_ENUMV(QUAD_4):
            return CellType::QUAD;
        default:
            fatal("unsupported CGNS element type " + std::to_string(static_cast<int>(e)) +
                  " (MIXED and high-order types are not supported)");
    }
}

}  // namespace

RawMesh* read_cgns_parallel(const std::string& path) {
    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Open the file in parallel on the whole communicator.
    cgp_mpi_comm(MPI_COMM_WORLD);
    cgp_pio_mode(CGP_INDEPENDENT);
    int fn = 0;
    if (cgp_open(path.c_str(), CG_MODE_READ, &fn) != CG_OK)
        fatal(std::string("cgp_open: ") + cg_get_error());

    auto* m = new RawMesh();
    m->nprocs = nprocs;
    m->rank = rank;

    // --- Replicated metadata (small; every rank reads it itself) ---
    int nbases = 0;
    if (cg_nbases(fn, &nbases) != CG_OK || nbases != 1) fatal("expected exactly 1 base");
    const int B = 1;
    int celldim = 0, physdim = 0;
    char basename[33] = "";
    cg_base_read(fn, B, basename, &celldim, &physdim);
    if (celldim != 3 || physdim != 3) fatal("base must be 3D (CellDim=3, PhysDim=3)");

    int nzones = 0;
    cg_nzones(fn, B, &nzones);
    if (nzones != 1) fatal("expected exactly 1 zone");
    int Z = 1;
    char zonename[33];
    cgsize_t sizes[3];
    cg_zone_read(fn, B, Z, zonename, sizes);
    CGNS_ENUMT(ZoneType_t) zt;
    cg_zone_type(fn, B, Z, &zt);
    if (zt != CGNS_ENUMV(Unstructured)) fatal("zone must be Unstructured");
    m->n_nodes_g = sizes[0];

    log_stat("CGNS: base='%s' zone='%s' nodes=%lld", basename, zonename, m->n_nodes_g);

    // --- Element sections (keep the original S index for the cgp_* calls) ---
    int nsecs = 0;
    cg_nsections(fn, B, Z, &nsecs);
    for (int S = 1; S <= nsecs; ++S) {
        char secname[33];
        CGNS_ENUMT(ElementType_t) etype;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;
        cg_section_read(fn, B, Z, S, secname, &etype, &start, &end, &nbndry, &parent_flag);
        if (etype == CGNS_ENUMV(BAR_2) || etype == CGNS_ENUMV(BAR_3)) continue;
        CellType t = cgns_elem_to_type(etype);
        SectionMeta sm;
        sm.name = secname;
        sm.type = t;
        sm.start = start;
        sm.end = end;
        sm.sec_idx = S;
        if (is_volume_type(t))
            m->vol_secs.push_back(sm);
        else
            m->surf_secs.push_back(sm);
    }
    // Dense global cell numbering: volume sections in their id order.
    // After this step every rank holds a contiguous chunk of cells (by gid)
    // read directly from the file; sections are renumbered densely 0..n-1.
    std::sort(m->vol_secs.begin(), m->vol_secs.end(),
              [](const SectionMeta& a, const SectionMeta& b) { return a.start < b.start; });
    long long nc = 0;
    for (auto& s : m->vol_secs) {
        s.cell_offset = nc;
        nc += s.end - s.start + 1;
    }
    m->n_cells_g = nc;
    if (m->n_cells_g == 0) fatal("no volume cells in the file");
    log_stat("CGNS: volume sections: %zu, cells: %lld; boundary sections: %zu", m->vol_secs.size(),
             m->n_cells_g, m->surf_secs.size());

    // --- BCs (ZoneBC): PointList/PointRange, GridLocation = FaceCenter ---
    int nbocos = 0;
    cg_nbocos(fn, B, Z, &nbocos);
    for (int bc = 1; bc <= nbocos; ++bc) {
        char bcname[33];
        CGNS_ENUMT(BCType_t) btype;
        CGNS_ENUMT(PointSetType_t) ptype;
        cgsize_t npnts = 0, normallistsize = 0;
        int normalidx[3];
        CGNS_ENUMT(DataType_t) ndtype;
        int ndataset = 0;
        cg_boco_info(fn, B, Z, bc, bcname, &btype, &ptype, &npnts, normalidx, &normallistsize,
                     &ndtype, &ndataset);
        CGNS_ENUMT(GridLocation_t) loc;
        cg_boco_gridlocation_read(fn, B, Z, bc, &loc);
        if (loc != CGNS_ENUMV(FaceCenter)) {
            log_warn_rank("BC '%s': GridLocation != FaceCenter, skipped", bcname);
            continue;
        }
        std::vector<cgsize_t> pnts(npnts);
        if (npnts > 0) cg_boco_read(fn, B, Z, bc, pnts.data(), nullptr);

        BCMeta bm;
        bm.cgns_type = BCTypeName[btype];
        if (ptype == CGNS_ENUMV(PointRange) && npnts == 2) {
            for (cgsize_t e = pnts[0]; e <= pnts[1]; ++e) bm.eids.push_back(e);
        } else {
            for (cgsize_t i = 0; i < npnts; ++i) bm.eids.push_back(pnts[i]);
        }
        char fam[33] = "";
        if (cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK)
            if (cg_famname_read(fam) == CG_OK && fam[0] != '\0') bm.name = fam;
        if (bm.name.empty()) bm.name = bcname;
        m->bcs.push_back(bm);
    }
    for (auto& b : m->bcs) {
        m->patch_list.push_back({b.name, b.cgns_type});
        log_stat("CGNS: BC '%s' type=%s nfaces=%zu -> patch %zu", b.name.c_str(),
                 b.cgns_type.c_str(), b.eids.size(), m->patch_list.size() - 1);
    }

    // --- Boundary sections: parallel slices + keys for BC matching ---
    {
        // Replicated eid -> patch id map.
        std::unordered_map<long long, int32_t> eid2patch;
        for (size_t p = 0; p < m->bcs.size(); ++p)
            for (long long e : m->bcs[p].eids) eid2patch[e] = static_cast<int32_t>(p);

        for (const auto& s : m->surf_secs) {
            const long long sec_n = s.end - s.start + 1;
            const std::vector<long long> d = block_displ(sec_n, nprocs);
            const long long lo = d[rank], hi = d[rank + 1];  // half-open
            if (lo >= hi) continue;
            const int npt = kNodesPerType[static_cast<int>(s.type)];
            std::vector<cgsize_t> buf(static_cast<size_t>(hi - lo) * npt);
            cgsize_t rs = static_cast<cgsize_t>(s.start + lo);
            cgsize_t re = static_cast<cgsize_t>(s.start + hi - 1);
            if (cgp_elements_read_data(fn, B, Z, s.sec_idx, rs, re, buf.data()) != CG_OK)
                fatal(std::string("cgp_elements_read_data(surface): ") + cg_get_error());
            for (long long i = 0; i < hi - lo; ++i) {
                SurfElem se;
                for (int k = 0; k < npt; ++k)
                    se.key.v[k] = static_cast<int32_t>(buf[i * npt + k] - 1);
                std::sort(se.key.v, se.key.v + 4);
                se.eid = static_cast<int32_t>(s.start + lo + i);
                auto it = eid2patch.find(se.eid);
                se.patch = (it != eid2patch.end()) ? it->second : -1;
                m->surf_elems.push_back(se);
            }
        }
    }

    // --- Distributions (block, for cells and for nodes) ---
    m->cell_displ = block_displ(m->n_cells_g, nprocs);
    m->node_displ = block_displ(m->n_nodes_g, nprocs);

    // --- Parallel read of the volume-cell connectivity (own slices) ---
    const long long nl = m->n_local();
    m->cgid.assign(nl, 0);
    m->ctype.assign(nl, 0);
    m->cnodes.assign(nl * 8, -1);  // slots sized for the worst case (HEXA), padded with -1

    long long written = 0;
    for (const auto& s : m->vol_secs) {
        const long long sec_n = s.end - s.start + 1;
        const long long cb = m->cell_displ[rank], ce = m->cell_displ[rank + 1];
        const long long lo = std::max(s.cell_offset, cb);
        const long long hi = std::min(s.cell_offset + sec_n, ce);  // half-open
        if (lo >= hi) continue;
        const int npt = kNodesPerType[static_cast<int>(s.type)];
        std::vector<cgsize_t> buf(static_cast<size_t>(hi - lo) * npt);
        cgsize_t rs = static_cast<cgsize_t>(s.start + (lo - s.cell_offset));
        cgsize_t re = static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset));
        if (cgp_elements_read_data(fn, B, Z, s.sec_idx, rs, re, buf.data()) != CG_OK)
            fatal(std::string("cgp_elements_read_data: ") + cg_get_error());
        for (long long i = 0; i < hi - lo; ++i) {
            m->cgid[written] = static_cast<int32_t>(lo + i);
            m->ctype[written] = static_cast<uint8_t>(s.type);
            int32_t* dst = &m->cnodes[written * 8];
            for (int k = 0; k < npt; ++k)
                dst[k] = static_cast<int32_t>(buf[i * npt + k] - 1);  // 1-based -> 0-based
            ++written;
        }
    }
    if (written != nl)
        fatal("cells read " + std::to_string(written) + ", expected " + std::to_string(nl));

    // --- Coordinates of this rank's node slice (C: 1=X, 2=Y, 3=Z) ---
    const long long nb = m->my_node_begin(), ne = m->my_node_end();
    const long long nmy = ne - nb;
    m->my_node_coords.assign(nmy * 3, 0.0);
    if (nmy > 0) {
        cgsize_t rs = static_cast<cgsize_t>(nb + 1);  // CGNS 1-based
        cgsize_t re = static_cast<cgsize_t>(ne);
        for (int dir = 0; dir < 3; ++dir) {
            std::vector<double> c(nmy);
            if (cgp_coord_read_data(fn, B, Z, dir + 1, &rs, &re, c.data()) != CG_OK)
                fatal(std::string("cgp_coord_read_data: ") + cg_get_error());
            for (long long i = 0; i < nmy; ++i) m->my_node_coords[3 * i + dir] = c[i];
        }
    }

    cgp_close(fn);

    log_stat("CGNS read: cells per rank min=%lld max=%lld, nodes on this rank: %lld", ll_min(nl),
             ll_max(nl), nmy);
    return m;
}

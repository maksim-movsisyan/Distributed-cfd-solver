# PARALLEL_CFD — Stage 1: Parallel Mesh Preprocessing

Parallel preprocessing toolkit for an unstructured-mesh CFD solver (compressible
viscous flow, Navier–Stokes). `mesh_partition` reads a CGNS mesh in parallel
(PCGNS / parallel HDF5) on all ranks, builds the face–cell connectivity in a
distributed fashion, partitions the mesh (dKaMinPar), constructs a one-hop
ghost layer (face neighbours + vertex-sharing neighbours for least-squares
gradients), computes the finite-volume geometry, and writes a single custom
HDF5 file with everything the solver needs, plus a text boundary-condition
template and VTU visualization of the partitioning.

`solver_mesh_check` reads that file exactly the way the solver will (each rank
reads its own hyperslab), verifies consistency (ghost-map handshake with
neighbours, volume sums, BC coverage) and prints detailed metadata.

## Project layout

    include/cfd/            public headers (header-only separation from src)
      mpi/                  logging + MPI helpers (all-to-all, reductions)
      mesh/                 element tables, raw mesh, faces, local mesh,
                            geometry, boundary matching
      partition/            dKaMinPar wrapper + SFC part-to-rank mapping
      io/cgns/              (headers shared with mesh data structures)
      io/solver_mesh/       custom HDF5 writer/loader utilities
      io/vtk/               VTU/PVTU output
    src/                    library sources mirroring include/
    apps/                   executables (mesh_partition, solver_mesh_check)
    tests/                  small runtime tests (dKaMinPar API smoke test)
    mesh/                   local test meshes (not committed)

Library structure (CMake targets):

- `cfd_mpi` — MPI wrappers, logging
- `cfd_mesh` — topology, ghost layer, renumbering, geometry, BC matching
- `cfd_partition` — dKaMinPar + space-filling-curve mapping of parts to ranks
- `cfd_cgns_io` — parallel CGNS reader (PCGNS `cgp_*` API)
- `cfd_solver_mesh_io` — custom HDF5 writer/loader + VTK export

## Prerequisites

GCC (C++20), OpenMPI, parallel HDF5, CGNS built with parallel I/O, KaMinPar,
TBB, CMake >= 3.28, Ninja. Full environment setup (WSL2 Ubuntu):
see [INSTALL-AND-SETUP-EN.md](INSTALL-AND-SETUP-EN.md).

## Build

    cmake --preset release
    cmake --build --preset release -j4

(presets `debug` / `release`; binaries appear in `build/release/`).

## Run

    mkdir -p out
    mpirun -np 4 build/release/mesh_partition mesh/DesktopTest.cgns \
        out/part.h5 --verbose --bc out/bc.txt --vtu out/vtu
    mpirun -np 4 build/release/solver_mesh_check out/part.h5 \
        --verbose --dump-vtu out/vtu/loaded

Options: `--verbose/-v` (statistics), `-vv` (extra detail), `--bc <file>`
(boundary condition template), `--vtu <dir>` (partitioning visualization).
On a 4-rank run each rank writes `part_XXXXX.vtu` plus a `part.pvtu`
collection; open the `.pvtu` in ParaView and colour by `rank` / `ghost` /
`patch`.

## Output file format (v2, flat parallel layout)

Global attributes (rank count, global cell/face/node counts, bounding box,
total volume) and per-rank offset datasets (`cells_off`, `nodes_off`,
`faces_off`, `nb_off`, `recv_off`, `send_off`, `patchfaces_off`, `n_own`),
followed by global arrays:

    cells/{type,gid,donor,nodes,centroid,volume}
    nodes/{xyz,gid}
    faces/{owner,neigh,type,nodes,centroid,normal,area,patch,donor}
    comm/{nb_ranks,recv_ghost_local,send_owned_local,recv_seg,send_seg}
    patches/face_idx

Data of rank *r* occupies the range `[off[r], off[r+1])` of every array.
Writing is two-phase: rank 0 creates the skeleton with full zero-fill
(serial driver), then all ranks write their hyperslabs simultaneously with
collective MPI-IO — the canonical parallel HDF5 pattern.

## Status

Validated on a 4800-cell hexahedral wedge mesh on 1/2/4 ranks: both
executables exit with rc=0, ghost-map handshake passes, the total volume
matches the analytic value, and all boundary faces are covered by BC patches.

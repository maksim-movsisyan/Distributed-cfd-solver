# cfd_linalg — distributed linear algebra module

Self-contained MPI library for the implicit-solution stage of the CFD solver.
Dependencies: **MPI + `cfd/core/types.hpp` only** — the module knows nothing
about meshes, halo exchangers, fields or residuals, and can be lifted out of
the project as a standalone library.

Public headers live in `include/cfd/linalg/` (umbrella: `cfd/linalg/linalg.hpp`).

## Architecture

```
LinearOperator (abstract)      Preconditioner (abstract)      IterativeSolver (abstract)
├── CsrMatrix   (bs = 1)       ├── SgsPreconditioner          └── BiCGSTAB
└── BsrMatrix   (bs = n_vars)  │   (CSR + BSR, hybrid)            (right-preconditioned)
                                └── IdentityPreconditioner
Vector / VectorLayout — ownership ranges, ghost exchange plan, fused reductions
```

The virtual seam sits at kernel granularity (one virtual call per SpMV /
preconditioner application); everything inside the kernels is raw pointers
over local arrays, with `CFD_RESTRICT` and compile-time block sizes.

## Distribution model

* Rows are distributed with **contiguous ownership**: rank r owns global rows
  `[begin_r, begin_r + n_local_r)`. An application maps local cells to rows
  with a simple offset (`global_row = local_offset + local_cell`).
* A vector is one contiguous array `[owned | ghosts]`. Matrices are assembled
  with global row/column ids; `assemble()` derives the ghost set from the
  referenced foreign columns, builds the point-to-point exchange plan and
  **localizes all column indices**, so SpMV and Gauss–Seidel sweeps never touch
  global ids.
* `Vector::updateGhosts()` is a single non-blocking round (pack → Irecv/Isend →
  Waitall → unpack) with per-layout scratch buffers.

### Contracts

| Component   | Entry                        | Exit                          |
| ----------- | ---------------------------- | ----------------------------- |
| `LinearOperator::apply(x, y)` | x ghosts current, x ≠ y | y owned written, y ghosts untouched |
| `Preconditioner::apply(r, z)` | r ghosts never read     | z owned written, **z ghosts current** |
| `IterativeSolver::solve`      | x/b ghosts may be stale | x holds the solution          |

Solvers call `updateGhosts()` exactly where the contracts require it — for
BiCGSTAB+SGS that is 2 halo rounds per iteration (inside the preconditioner).

## Assembly workflow (time stepping in mind)

```cpp
BsrMatrix A(comm, n_global_block_rows, n_local_block_rows, n_vars);
for each cell contribution: A.addBlock(grow, gcol, block5x5);  // duplicates sum
A.assemble();              // structure + ghost plan + local cols (collective)

// later time steps, same sparsity:
for each contribution: A.addBlock(...);
A.assembleValues();        // binary-search insertion, local, no global sort
```

## Performance notes

* BiCGSTAB: **2 MPI_Allreduce rounds per iteration** — `r0·v` plus a single
  fused 5-way inner product that yields omega, rho_{k+1} and ‖r_{k+1}‖
  (`Vector::batchedDots`). The monitored norm is refreshed from the true
  recurrence vector every 32 iterations.
* BSR kernels are specialized for block sizes 1–8 (dispatch in
  `src/linalg/bsr_kernels.hpp`); bs = 5 — the compressible-NS case — runs a
  fully unrolled inner loop. Larger blocks use a dynamic fallback.
* SGS inverts/factorizes diagonals once at `setup()`; each apply is two
  sweeps + two halo rounds (hybrid distributed SGS).
* Solver workspace vectors are reused across solves for a fixed layout.

## Limitations / roadmap

* Single-threaded per rank (MPI-only); OpenMP hybrid is a natural extension.
* Halos in SpMV are refreshed before the kernel, not overlapped with interior
  compute — the exchange API is structured so a begin/end split can be added.
* Roadmap candidates: ILU(0), block Jacobi with additive overlap, multicolor
  Gauss–Seidel, pipelined BiCGSTAB (1 reduction/iter), GMRES.

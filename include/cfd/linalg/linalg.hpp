// Umbrella header for the distributed linear algebra module (cfd_linalg).
//
// Self-contained: MPI + cfd/core/types.hpp only — no mesh, halo-exchanger or
// solver-level dependencies. See src/linalg/README.md for the architecture.
#pragma once

#include "cfd/linalg/bicgstab.hpp"
#include "cfd/linalg/bsr_matrix.hpp"
#include "cfd/linalg/csr_matrix.hpp"
#include "cfd/linalg/operator.hpp"
#include "cfd/linalg/preconditioner.hpp"
#include "cfd/linalg/preconditioners.hpp"
#include "cfd/linalg/solver.hpp"
#include "cfd/linalg/types.hpp"
#include "cfd/linalg/vector.hpp"
#include "cfd/linalg/vector_layout.hpp"
